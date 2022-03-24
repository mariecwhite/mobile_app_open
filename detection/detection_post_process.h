/* Copyright 2022 The MLPerf Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MLPERF_DETECTION_DETECTION_POST_PROCESS_H_
#define MLPERF_DETECTION_DETECTION_POST_PROCESS_H_

#include <cstdint>
#include <vector>

#include "detection/center_size_encoded_boxes.pb.h"

// Options for the QuantizedPostProcess class.
struct PostProcessOptions {
  // The height of the input image.
  int input_height = 0;
  // The width of the input image.
  int input_width = 0;
  // The number of classes detected by the model, including the background
  // class.
  int num_classes = 91;
  // The offset of the most meaningful class. Set this to one if there is a
  // background class.
  int class_offset = 1;
  // The maximum number of detections to output per query.
  int max_detections = 50;
  // The minimum class score used when performing NMS.
  float nms_score_threshold = 0.4f;
  // The minimum IoU overlap used when performing NMS.
  float nms_iou_threshold = 0.6f;
  // The anchor configuration.
  mlperf::mobile::detection::CenterSizeEncodedBoxes anchors;
};

// The top left (x1, y1) and bottom right (x2, y2) coordinates of a bounding
// box.
struct BoxCoords {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
};

// Represents a detection.
struct Detection {
  BoxCoords box_coords;
  int class_index = 0;
  float class_score = 0.0f;
};

// Parameters used to convert a quantized variable to floating point.
struct QuantizationParams {
  int zero_point = 0;
  float scale = 0.0f;
};

// Metadata about an output tensor.
struct QuantizedOutput {
  // Pointer to the data.
  uint8_t* data;
  QuantizationParams quant_params;
  int num_boxes = 0;
};

class DetectionPostProcess {
 public:
  DetectionPostProcess(const PostProcessOptions& options);

  // Post-processes `output_locations` and `output_scores` based on
  // post-process options and returns the highest confidence detections.
  // `output_locations`: array of size `[1, NUM_BOXES, 4]`, where 4 represents
  //     the box encoding in `[y_center, x_center, height, width]`.
  // `output_scores`: array of size `[1, NUM_BOXES, NUM_CLASSES]`.
  std::vector<Detection> Run(const float* box_encodings,
                             const float* class_predictions,
                             int num_boxes);

  // Post-processes the location and score tensors of a quantized SSD model.
  // Expects a model to have output tensors representing quantized location and
  // score values, usually in alternating order. For example, a 4 layer
  // quantized SSD model would have the output tensors below.
  //
  // e.g. BoxPredictor_0/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,
  //      BoxPredictor_0/ClassPredictor/act_quant/FakeQuantWithMinMaxVars,
  //      BoxPredictor_1/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,
  //      BoxPredictor_1/ClassPredictor/act_quant/FakeQuantWithMinMaxVars,
  //      BoxPredictor_2/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,
  //      BoxPredictor_2/ClassPredictor/act_quant/FakeQuantWithMinMaxVars,
  //      BoxPredictor_3/BoxEncodingPredictor/act_quant/FakeQuantWithMinMaxVars,
  //      BoxPredictor_3/ClassPredictor/act_quant/FakeQuantWithMinMaxVars
  std::vector<Detection> Run(const std::vector<QuantizedOutput>& quant_box_encodings,
                             const std::vector<QuantizedOutput>& quant_class_predictions);

 protected:
  BoxCoords* DecodeCenterSizeBoxes(const float* box_encodings, int num_boxes);

  PostProcessOptions options_;
  std::vector<BoxCoords> decoded_anchor_boxes_;
};

#endif  // MLPERF_DETECTION_DETECTION_POST_PROCESS_H_
