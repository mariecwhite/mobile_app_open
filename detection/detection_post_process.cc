#include "detection/detection_post_process.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

#include "detection/center_size_encoded_boxes.pb.h"

namespace {
// The default scale for anchor y-coordinate.
const float kBoxCoderYScale = 10.0;
// The default scale for anchor x-coordinate.
const float kBoxCoderXScale = 10.0;
// The default scale for anchor width.
const float kBoxCoderWidthScale = 5.0;
// The default scale for anchor height.
const float kBoxCoderHeightScale = 5.0;

// The logistic function.
inline float Logit(float val) { return -std::log(1.0f / val - 1.0f); }
inline float InverseLogit(float val) {
  return 1.0f / (1.0f + std::exp(-1 * val));
}

inline BoxCoords DecodeBox(float box_y_center,
                           float box_x_center,
                           float box_height,
                           float box_width,
                           float anchor_y,
                           float anchor_x,
                           float anchor_height,
                           float anchor_width) {
  float y_center = box_y_center / kBoxCoderYScale * anchor_height + anchor_y;
  float x_center = box_x_center / kBoxCoderXScale * anchor_width + anchor_x;
  float h = std::exp(box_height / kBoxCoderHeightScale) * anchor_height;
  float w = std::exp(box_width / kBoxCoderWidthScale) * anchor_width;
  BoxCoords coords;
  coords.x1 = x_center - w / 2;
  coords.y1 = y_center - h / 2;
  coords.x2 = x_center + w / 2;
  coords.y2 = y_center + h / 2;
  return coords;
}

inline float ComputeArea(const BoxCoords& box) {
  const float width = box.x2 - box.x1;
  const float height = box.y2 - box.y1;
  if (width <= 0 || height < 0) {
    return 0;
  }
  return width * height;
}

inline float ComputeIou(const BoxCoords& box1, const BoxCoords& box2) {
  const float area1 = ComputeArea(box1);
  const float area2 = ComputeArea(box2);
  if (area1 <= 0 || area2 <= 0) return 0.0;
  BoxCoords intersection_box;
  intersection_box.x1 = std::max(box1.x1, box2.x1);
  intersection_box.y1 = std::max(box1.y1, box2.y1);
  intersection_box.x2 = std::min(box1.x2, box2.x2);
  intersection_box.y2 = std::min(box1.y2, box2.y2);
  const float intersection_area = ComputeArea(intersection_box);
  return intersection_area / (area1 + area2 - intersection_area);
}

inline float Dequantize(uint8_t x, const QuantizationParams& quant_params) {
  return (static_cast<float>(x) - quant_params.zero_point) * quant_params.scale;
}

inline uint8_t Quantize(float x, const QuantizationParams& quant_params) {
  return x / quant_params.scale + quant_params.zero_point;
}

}  // namespace

DetectionPostProcess::DetectionPostProcess(const PostProcessOptions& options)
    : options_(options) {
}

BoxCoords* DetectionPostProcess::DecodeCenterSizeBoxes(const float* box_encodings,
                                                       int num_boxes) {
  decoded_anchor_boxes_.resize(num_boxes);
  const float* box_encoding_ptr = box_encodings;
  for (int i = 0; i < num_boxes; ++i) {
    // Boxes are encoded as [y_center, x_center, height, width].
    decoded_anchor_boxes_[i] =
        DecodeBox(box_encoding_ptr[0], box_encoding_ptr[1],
                  box_encoding_ptr[2], box_encoding_ptr[3],
                  options_.anchors.y(i), options_.anchors.x(i),
                  options_.anchors.h(i), options_.anchors.w(i));
    box_encoding_ptr += 4;
  }
  return &decoded_anchor_boxes_[0];
}

std::vector<Detection> DetectionPostProcess::Run(const float* box_encodings,
                                                 const float* class_predictions,
                                                 int num_boxes) {
  BoxCoords* box_ptr = DecodeCenterSizeBoxes(box_encodings, num_boxes);
  const float* scores_ptr = class_predictions;
  int effective_num_classes = options_.num_classes - options_.class_offset;
  std::vector<Detection> detections;

  // At each box location, take the highest confidence detection.
  for (int i = 0; i < num_boxes; ++i) {
    scores_ptr += options_.class_offset;
    float max_score = 0;
    int max_class_index = -1;
    for (int k = 0; k < effective_num_classes; ++k) {
      if (*scores_ptr > max_score) {
        max_score = *scores_ptr;
        max_class_index = k;
      }
      scores_ptr++;
    }
    if (max_score >= options_.nms_score_threshold && max_class_index != -1) {
      Detection detection;
      detection.box_coords = box_ptr[i];
      detection.class_index = max_class_index;
      detection.class_score = max_score;
      detections.push_back(detection);
    }
  }

  // Perform nms on detections.
  std::vector<Detection> nms_detections;
  std::vector<int> indices(detections.size());
  for (int i = 0; i < detections.size(); ++i) {
    indices[i] = i;
  }
  std::sort(indices.begin(), indices.end(),
            [&detections](const int i, const int j) {
              return detections[i].class_score > detections[j].class_score;
            });
  for (int i: indices) {
    bool keep_index = true;
    for (const Detection& nms_detection: nms_detections) {
      if (ComputeIou(detections[i].box_coords, nms_detection.box_coords) >=
          options_.nms_iou_threshold) {
        keep_index = false;
        break;
      }
    }
    if (keep_index) {
      nms_detections.push_back(detections[i]);
    }
    if (nms_detections.size() >= options_.max_detections) {
      break;
    }
  }
  return nms_detections;
}

std::vector<Detection> DetectionPostProcess::Run(
    const std::vector<QuantizedOutput>& quant_box_encodings,
    const std::vector<QuantizedOutput>& quant_class_predictions) {
  int effective_num_classes = options_.num_classes - options_.class_offset;
  std::vector<Detection> detections;
  int anchors_index = 0;

  // At each box location, take the highest confidence detection.
  for (int layer_index = 0; layer_index < quant_box_encodings.size();
       ++layer_index) {
    // Represent the score threshold as quantized logits to do comparisons with
    // the quantized scores.
    const QuantizationParams& score_quant_params =
        quant_class_predictions[layer_index].quant_params;
    const uint8_t score_threshold_logit = Quantize(
        Logit(options_.nms_score_threshold), score_quant_params);

    const QuantizationParams& box_quant_params =
        quant_box_encodings[layer_index].quant_params;
    const uint8_t* box_ptr = quant_box_encodings[layer_index].data;
    const uint8_t* scores_ptr = quant_class_predictions[layer_index].data;

    for (int i = 0; i < quant_box_encodings[layer_index].num_boxes; ++i) {
      scores_ptr += options_.class_offset;
      uint8_t max_score = 0;
      int max_class_index = -1;
      for (int k = 0; k < effective_num_classes; ++k) {
        if (*scores_ptr > max_score) {
          max_score = *scores_ptr;
          max_class_index = k;
        }
        scores_ptr++;
      }
      if (max_score >= score_threshold_logit && max_class_index != -1) {
        Detection detection;
        // Decode box.
        detection.box_coords = DecodeBox(
            Dequantize(box_ptr[0], box_quant_params),
            Dequantize(box_ptr[1], box_quant_params),
            Dequantize(box_ptr[2], box_quant_params),
            Dequantize(box_ptr[3], box_quant_params),
            options_.anchors.y(anchors_index),
            options_.anchors.x(anchors_index),
            options_.anchors.h(anchors_index),
            options_.anchors.w(anchors_index));
        // Decode score.
        detection.class_score =
            InverseLogit(Dequantize(max_score, score_quant_params));
        detection.class_index = max_class_index;
        detections.push_back(detection);
      }
      anchors_index++;
      box_ptr += 4;
    }
  }

  // Perform nms on detections.
  std::vector<Detection> nms_detections;
  std::vector<int> indices(detections.size());
  for (int i = 0; i < detections.size(); ++i) {
    indices[i] = i;
  }
  std::sort(indices.begin(), indices.end(),
            [&detections](const int i, const int j) {
              return detections[i].class_score > detections[j].class_score;
            });
  for (int i: indices) {
    bool keep_index = true;
    for (const Detection& nms_detection: nms_detections) {
      if (ComputeIou(detections[i].box_coords, nms_detection.box_coords) >=
          options_.nms_iou_threshold) {
        keep_index = false;
        break;
      }
    }
    if (keep_index) {
      nms_detections.push_back(detections[i]);
    }
    if (nms_detections.size() >= options_.max_detections) {
      break;
    }
  }
  return nms_detections;
}
