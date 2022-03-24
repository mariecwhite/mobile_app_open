/* Copyright 2020-2021 The MLPerf Authors. All Rights Reserved.

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
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "android/cpp/c/backend_c.h"
#include "android/cpp/c/type.h"
#include "detection/detection_post_process.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"

#include "iree_settings.h"

struct IreeBackendData {
  const char* name = "IREE";
  const char* vendor = "Google";

  iree_runtime_instance_t* iree_runtime_instance = nullptr;
  iree_runtime_session_options_t session_options;
  iree_runtime_session_t* iree_runtime_session = nullptr;
  iree_runtime_call_t iree_runtime_call;
  bool is_detection = false;
  // This flag is only relevant for detection models.
  bool is_quantized = false;

  // Input buffers.
  int32_t input_count = 0;
  std::vector<iree_hal_buffer_view_t*> input_buffers;
  std::vector<mlperf_data_t> input_types;
  std::vector<std::vector<int>> input_dims;

  // Outputs.
  int32_t output_count = 0;
  std::vector<iree_hal_buffer_t*> output_buffers;
  std::vector<mlperf_data_t> output_types;
  std::vector<std::vector<int>> output_dims;

  int32_t num_queries = 0;

  // Detection post process.
  DetectionPostProcess* post_process;
  std::vector<QuantizedOutput> quant_box_encodings;
  std::vector<QuantizedOutput> quant_class_predictions;

  // The bounding box coordinates of the detections.
  // Max size is |max_detections| * |box_code_size|.
  float* detection_boxes{nullptr};
  uint32_t detection_boxes_size;
  // The class of each detection. Max size is |max_detections|.
  float* detection_classes{nullptr};
  uint32_t detection_classes_size;
  // The confidence scores of each detection. Max size is |max_detections|.
  float* detection_scores{nullptr};
  uint32_t detection_scores_size;
  // The number of detections found in the image.
  float num_detections = 0.0f;
};

static bool backendExists = false;

inline mlperf_data_t::Type IreeType2Type(iree_hal_element_type_t type) {
  switch (type) {
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:return mlperf_data_t::Float32;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:return mlperf_data_t::Uint8;
    case IREE_HAL_ELEMENT_TYPE_INT_8:return mlperf_data_t::Int8;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:return mlperf_data_t::Float16;
    case IREE_HAL_ELEMENT_TYPE_INT_32:return mlperf_data_t::Int32;
    case IREE_HAL_ELEMENT_TYPE_INT_64:return mlperf_data_t::Int64;
    default:printf("IREE type %d not supported", type);
      return mlperf_data_t::Float32;
  }
}

bool iree_map_output(IreeBackendData* backend_data, int32_t index,
                     void** data) {
  iree_vm_list_t* output_list =
      iree_runtime_call_outputs(&backend_data->iree_runtime_call);
  iree_vm_ref_t value = {0};
  iree_status_t
      status = iree_vm_list_get_ref_assign(output_list, index, &value);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return false;
  }

  iree_hal_buffer_view_t* buffer_view = nullptr;
  status = iree_hal_buffer_view_check_deref(value, &buffer_view);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return false;
  }

  iree_device_size_t byte_offset = 0;
  iree_device_size_t byte_length = IREE_WHOLE_BUFFER;
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_hal_buffer_mapping_t buffer_mapping;
  status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, byte_offset,
      byte_length, &buffer_mapping);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return false;
  }

  if (backend_data->output_buffers[index]) {
    iree_hal_buffer_release(backend_data->output_buffers[index]);
    backend_data->output_buffers[index] = nullptr;
  }
  backend_data->output_buffers[index] = buffer;
  iree_hal_buffer_retain(backend_data->output_buffers[index]);
  *data = buffer_mapping.contents.data;
  return true;
}

template<typename T>
std::vector<T> split(const std::string& input, char token) {
  std::vector<T> result;
  std::stringstream ss(input);
  T i;
  while (ss >> i) {
    result.push_back(i);
    if (ss.peek() == token) {
      ss.ignore();
    }
  }
  return result;
}

void cleanup(IreeBackendData* backend_data) {
  if (backend_data) {
    for (auto* buffer_view: backend_data->input_buffers) {
      iree_hal_buffer_view_release(buffer_view);
    }
    for (auto* buffer: backend_data->output_buffers) {
      iree_hal_buffer_release(buffer);
    }
    iree_runtime_call_deinitialize(&backend_data->iree_runtime_call);
    iree_runtime_session_release(backend_data->iree_runtime_session);
    iree_runtime_instance_release(backend_data->iree_runtime_instance);
    delete backend_data;
    backendExists = false;
  }
}

// TFLite is the standard backend for all hardwares.
bool mlperf_backend_matches_hardware(const char** not_allowed_message,
                                     const char** settings,
                                     const mlperf_device_info_t* device_info) {
  *not_allowed_message = nullptr;
  *settings = iree_settings.c_str();
  printf("IREE runtime matches hardware\n");
  return true;
}

// Create a new backend and return the pointer to it.
mlperf_backend_ptr_t mlperf_backend_create(
    const char* model_path, mlperf_backend_configuration_t* configs,
    const char* native_lib_path) {
  // Verify only one instance of the backend exists at any time
  if (backendExists) {
    printf("Error: Only one backend instance should exist at a time");
    return nullptr;
  }

  // Retrieve config.
  iree_string_view_t driver = iree_string_view_empty();
  iree_string_view_t entry_function = iree_string_view_empty();
  iree_string_view_t function_inputs = iree_string_view_empty();
  iree_string_view_t function_outputs = iree_string_view_empty();
  std::string function_output_scales;
  std::string function_output_zero_points;
  for (int i = 0; i < configs->count; ++i) {
    if (strcmp(configs->keys[i], "driver") == 0) {
      driver = iree_make_cstring_view(configs->values[i]);
    } else if (strcmp(configs->keys[i], "entry_function") == 0) {
      entry_function = iree_make_cstring_view(configs->values[i]);
    } else if (strcmp(configs->keys[i], "function_inputs") == 0) {
      function_inputs = iree_make_cstring_view(configs->values[i]);
    } else if (strcmp(configs->keys[i], "function_outputs") == 0) {
      function_outputs = iree_make_cstring_view(configs->values[i]);
    } else if (strcmp(configs->keys[i], "function_output_scales") == 0) {
      function_output_scales = configs->values[i];
    } else if (strcmp(configs->keys[i], "function_output_zero_points") == 0) {
      function_output_zero_points = configs->values[i];
    }
  }

  IreeBackendData* backend_data = new IreeBackendData();
  backendExists = true;
  backend_data->is_detection = configs->is_detection;

  // Create IREE runtime.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_status_t status =
      iree_runtime_instance_create(&instance_options, iree_allocator_system(),
                                   &backend_data->iree_runtime_instance);

  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    cleanup(backend_data);
    return nullptr;
  }

  // Create IREE runtime session.
  iree_hal_device_t* device = nullptr;
  status = iree_runtime_instance_try_create_default_device(
      backend_data->iree_runtime_instance, driver, &device);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    cleanup(backend_data);
    return nullptr;
  }

  iree_runtime_session_options_initialize(&backend_data->session_options);
  status = iree_runtime_session_create_with_device(
      backend_data->iree_runtime_instance,
      &backend_data->session_options,
      device,
      iree_runtime_instance_host_allocator(backend_data->iree_runtime_instance),
      &backend_data->iree_runtime_session);
  iree_hal_device_release(device);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    cleanup(backend_data);
    return nullptr;
  }

  // Load module.
  status =
      iree_runtime_session_append_bytecode_module_from_file(backend_data->iree_runtime_session,
                                                            model_path);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    cleanup(backend_data);
    return nullptr;
  }

  // Initialize the call function.
  status = iree_runtime_call_initialize_by_name(
      backend_data->iree_runtime_session, entry_function,
      &backend_data->iree_runtime_call);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    cleanup(backend_data);
    return nullptr;
  }

  // Get input/output metadata.
  iree_vm_function_t main;
  status = iree_runtime_session_lookup_function(
      backend_data->iree_runtime_session, entry_function, &main);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    cleanup(backend_data);
    return nullptr;
  }
  iree_vm_function_signature_t
      main_signature = iree_vm_function_signature(&main);
  iree_string_view_t arguments, results;
  status = iree_vm_function_call_get_cconv_fragments(
      &main_signature, &arguments, &results);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    cleanup(backend_data);
    return nullptr;
  }
  backend_data->input_count = (int32_t) arguments.size;
  backend_data->input_buffers.resize(backend_data->input_count);
  backend_data->input_types.resize(backend_data->input_count);
  backend_data->input_dims.resize(backend_data->input_count);
  backend_data->output_count = (int32_t) results.size;
  backend_data->output_buffers.resize(backend_data->output_count);
  backend_data->output_types.resize(backend_data->output_count);
  backend_data->output_dims.resize(backend_data->output_count);

  // Create input buffers.
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(backend_data->iree_runtime_session);

  for (iree_host_size_t i = 0; i < backend_data->input_count; ++i) {
    iree_string_view_t input = iree_string_view_empty();
    iree_string_view_split(function_inputs, ',', &input, &function_inputs);

    iree_hal_buffer_view_t* buffer_view = nullptr;
    status = iree_hal_buffer_view_parse(input, device_allocator, &buffer_view);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      cleanup(backend_data);
      return nullptr;
    }
    status = iree_runtime_call_inputs_push_back_buffer_view(
        &backend_data->iree_runtime_call, buffer_view);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      cleanup(backend_data);
      return nullptr;
    }
    backend_data->input_buffers[i] = buffer_view;
    backend_data->input_types[i].type =
        IreeType2Type(iree_hal_buffer_view_element_type(buffer_view));
    backend_data->input_types[i].size =
        iree_hal_buffer_view_byte_length(buffer_view)
            / iree_hal_buffer_view_element_size(buffer_view);

    // Create dims.
    std::vector<int> dims(iree_hal_buffer_view_shape_rank(buffer_view));
    const iree_hal_dim_t
        * dims_ptr = iree_hal_buffer_view_shape_dims(buffer_view);
    for (int j = 0; j < dims.size(); ++j) {
      dims[j] = dims_ptr[j];
    }
    backend_data->input_dims[i] = dims;
  }

  // Set output type.
  for (iree_host_size_t i = 0; i < backend_data->output_count; ++i) {
    iree_string_view_t output = iree_string_view_empty();
    iree_string_view_split(function_outputs, ',', &output, &function_outputs);

    iree_string_view_t shape_str = iree_string_view_empty();
    iree_string_view_t type_str = iree_string_view_empty();

    iree_host_size_t last_x_index = iree_string_view_find_last_of(
        output, IREE_SV("x"), IREE_STRING_VIEW_NPOS);

    shape_str = iree_string_view_substr(output, 0, last_x_index);
    iree_host_size_t shape_rank = 0;
    status = iree_hal_parse_shape(shape_str, 0, NULL, &shape_rank);
    iree_hal_dim_t* shape =
        (iree_hal_dim_t*) iree_alloca(shape_rank * sizeof(iree_hal_dim_t));
    status = iree_hal_parse_shape(shape_str, shape_rank, shape, &shape_rank);

    int32_t element_size = 1;
    std::vector<int> dims(shape_rank);
    for (int j = 0; j < shape_rank; ++j) {
      element_size *= shape[j];
      dims[j] = shape[j];
    }
    backend_data->output_dims[i] = dims;

    type_str = iree_string_view_substr(output, last_x_index + 1,
                                       IREE_STRING_VIEW_NPOS);
    iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
    iree_hal_parse_element_type(type_str, &element_type);

    backend_data->output_types[i].type = IreeType2Type(element_type);
    backend_data->output_types[i].size = element_size;

    // If the output tensor is quantized, we consider the model is quantized.
    if (element_type == IREE_HAL_ELEMENT_TYPE_UINT_8) {
      backend_data->is_quantized = true;
    }
  }

  // Update postprocess.
  if (backend_data->is_detection) {
    PostProcessOptions post_process_options;
    post_process_options.input_height = backend_data->input_dims[0][1];
    post_process_options.input_width = backend_data->input_dims[0][2];

    std::vector<char> proto_bytes;
    std::ifstream infile;
    infile.open(configs->anchor_path, std::ios::binary | std::ios::ate);
    if (!infile.is_open()) {
      return nullptr;
    }
    proto_bytes.resize(infile.tellg());
    if (proto_bytes.empty()) {
      return nullptr;
    }
    infile.seekg(0);
    if (!infile.read(&proto_bytes[0], proto_bytes.size())) {
      infile.close();
      return nullptr;
    }
    infile.close();
    if (!post_process_options.anchors.ParseFromArray(proto_bytes.data(),
                                                     proto_bytes.size())) {
      return nullptr;
    }

    backend_data->post_process = new DetectionPostProcess(post_process_options);
    const int num_detected_boxes = post_process_options.max_detections;
    backend_data->detection_boxes_size = num_detected_boxes * 4;
    backend_data->detection_boxes =
        new float[backend_data->detection_boxes_size];
    backend_data->detection_classes_size = num_detected_boxes;
    backend_data->detection_classes =
        new float[backend_data->detection_classes_size];
    backend_data->detection_scores_size = num_detected_boxes;
    backend_data->detection_scores =
        new float[backend_data->detection_scores_size];

    if (backend_data->is_quantized) {
      // The number of output tensors is divided by 2 since one layer represents
      // a box and class tensor.
      int num_output_layers = backend_data->output_count / 2;
      backend_data->quant_box_encodings.resize(num_output_layers);
      backend_data->quant_class_predictions.resize(num_output_layers);

      std::vector<float> scales = split<float>(function_output_scales, ',');
      std::vector<int>
          zero_points = split<int>(function_output_zero_points, ',');
      assert(scales.size() == zero_points.size());
      assert(scales.size() == num_output_layers);

      for (int i = 0; i < num_output_layers; ++i) {
        const std::vector<int>
            & locations_dims = backend_data->output_dims[2 * i];
        const std::vector<int>
            & scores_dims = backend_data->output_dims[2 * i + 1];

        // Get number of boxes for each layer.
        int locations_num_boxes =
            locations_dims[0] * locations_dims[1] * locations_dims[2]
                * (locations_dims[3] / 4);
        int scores_num_boxes = scores_dims[0] * scores_dims[1] * scores_dims[2]
            * (scores_dims[3] / post_process_options.num_classes);

        assert(locations_num_boxes == scores_num_boxes);
        backend_data->quant_box_encodings[i].num_boxes = locations_num_boxes;
        backend_data->quant_class_predictions[i].num_boxes = scores_num_boxes;

        backend_data->quant_box_encodings[i].quant_params =
            {zero_points[2 * i], scales[2 * i]};
        backend_data->quant_class_predictions[i].quant_params =
            {zero_points[2 * i + 1], scales[2 * i + 1]};
      }
    }
  }

  printf("Backend successfully created.\n");
  return backend_data;
}

// Vendor name who create this backend.
const char* mlperf_backend_vendor_name(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  return backend_data->vendor;
}

// Return the name of this backend.
const char* mlperf_backend_name(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  return backend_data->name;
}

// Destroy the backend pointer and its data.
void mlperf_backend_delete(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;

  delete backend_data->post_process;
  delete[] backend_data->detection_boxes;
  delete[] backend_data->detection_classes;
  delete[] backend_data->detection_scores;

  printf("Number of queries issued: %d\n", backend_data->num_queries);
  cleanup(backend_data);
}

// Run the inference for a sample.
mlperf_status_t mlperf_backend_issue_query(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;

  iree_status_t status =
      iree_runtime_call_invoke(&backend_data->iree_runtime_call, /*flags=*/0);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MLPERF_FAILURE;
  }

  if (backend_data->is_detection) {
    std::vector<Detection> detections;
    if (backend_data->is_quantized) {
      for (int i = 0; i < backend_data->quant_box_encodings.size(); ++i) {
        // Get reference to box encodings.
        if (!iree_map_output(backend_data,
                             2 * i,
                             reinterpret_cast<void**>(&backend_data->quant_box_encodings[i].data))) {
          return MLPERF_FAILURE;
        }
        if (!iree_map_output(backend_data,
                             2 * i + 1,
                             reinterpret_cast<void**>(&backend_data->quant_class_predictions[i].data))) {
          return MLPERF_FAILURE;
        }
      }
      detections =
          backend_data->post_process->Run(backend_data->quant_box_encodings,
                                          backend_data->quant_class_predictions);
    } else {
      // Get reference to box encodings.
      void* box_encodings;
      if (!iree_map_output(backend_data, 0, &box_encodings)) {
        return MLPERF_FAILURE;
      }
      void* class_predictions;
      if (!iree_map_output(backend_data, 1, &class_predictions)) {
        return MLPERF_FAILURE;
      }
      const float num_boxes = backend_data->output_dims[0][1];
      detections =
          backend_data->post_process->Run(reinterpret_cast<float*>(box_encodings),
                                          reinterpret_cast<float*>(class_predictions),
                                          num_boxes);
    }
    backend_data->num_detections = detections.size();
    // Convert detections to output tensors.
    float* detected_boxes_ptr = &backend_data->detection_boxes[0];
    float* detected_classes_ptr = &backend_data->detection_classes[0];
    float* detected_scores_ptr = &backend_data->detection_scores[0];
    for (const Detection& nms_detection: detections) {
      // Convert box coords to COCO form: normalized (ymin, xmin, ymax, xmax).
      *detected_boxes_ptr = nms_detection.box_coords.y1;
      *(detected_boxes_ptr + 1) = nms_detection.box_coords.x1;
      *(detected_boxes_ptr + 2) = nms_detection.box_coords.y2;
      *(detected_boxes_ptr + 3) = nms_detection.box_coords.x2;
      detected_boxes_ptr += 4;
      *detected_classes_ptr = nms_detection.class_index;
      *detected_scores_ptr = nms_detection.class_score;
      detected_classes_ptr++;
      detected_scores_ptr++;
    }
  }

  ++backend_data->num_queries;
  return MLPERF_SUCCESS;
}

// Flush the staged queries immediately.
mlperf_status_t mlperf_backend_flush_queries(mlperf_backend_ptr_t backend_ptr) {
  return MLPERF_SUCCESS;
}

// Return the number of inputs of the model.
int32_t mlperf_backend_get_input_count(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  return backend_data->input_count;
}

// Return the type of the ith input.
mlperf_data_t mlperf_backend_get_input_type(mlperf_backend_ptr_t backend_ptr,
                                            int32_t i) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  return backend_data->input_types[i];
}

// Set the data for ith input.
mlperf_status_t mlperf_backend_set_input(mlperf_backend_ptr_t backend_ptr,
                                         int32_t batch_index, int32_t i,
                                         void* data) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  if (i >= backend_data->input_count) {
    printf("Invalid input index");
    return MLPERF_FAILURE;
  }
  iree_device_size_t byte_offset = 0;
  iree_device_size_t byte_length = IREE_WHOLE_BUFFER;
  iree_hal_buffer_view_t* buffer_view = backend_data->input_buffers[i];
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  iree_hal_buffer_mapping_t buffer_mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_WRITE, byte_offset,
      byte_length, &buffer_mapping);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MLPERF_FAILURE;
  }
  memcpy(buffer_mapping.contents.data,
         data,
         iree_hal_buffer_view_byte_length(buffer_view));
  return MLPERF_SUCCESS;
}

// Return the number of outputs for the model.
int32_t mlperf_backend_get_output_count(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  if (backend_data->is_detection) {
    return 4;
  }
  return backend_data->output_count;
}

// Return the type of ith output.
mlperf_data_t mlperf_backend_get_output_type(mlperf_backend_ptr_t backend_ptr,
                                             int32_t i) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  if (backend_data->is_detection) {
    switch (i) {
      case 0: {
        mlperf_data_t type;
        type.type = mlperf_data_t::Float32;
        type.size = backend_data->detection_boxes_size;
        return type;
      }
      case 1: {
        mlperf_data_t type;
        type.type = mlperf_data_t::Float32;
        type.size = backend_data->detection_classes_size;
        return type;
      }
      case 2: {
        mlperf_data_t type;
        type.type = mlperf_data_t::Float32;
        type.size = backend_data->detection_scores_size;
        return type;
      }
      default: {
        mlperf_data_t type;
        type.type = mlperf_data_t::Float32;
        type.size = 1;
        return type;
      }
    }
  } else {
    return backend_data->output_types[i];
  }
}

// Get the data from ith output.
mlperf_status_t mlperf_backend_get_output(mlperf_backend_ptr_t backend_ptr,
                                          uint32_t batch_index, int32_t i,
                                          void** data) {
  if (batch_index > 0) {
    printf("Only batch size 1 is currently supported.");
    return MLPERF_FAILURE;
  }
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  if (backend_data->is_detection) {
    switch (i) {
      case 0: {
        *data = backend_data->detection_boxes;
        break;
      }
      case 1: {
        *data = backend_data->detection_classes;
        break;
      }
      case 2: {
        *data = backend_data->detection_scores;
        break;
      }
      case 3: {
        *data = &backend_data->num_detections;
        break;
      }
      default: {
        printf("Index does not exist.");
        return MLPERF_FAILURE;
      }
    }
  } else {
    if (!iree_map_output(backend_data, i, data)) {
      return MLPERF_FAILURE;
    }
  }
  return MLPERF_SUCCESS;
}
