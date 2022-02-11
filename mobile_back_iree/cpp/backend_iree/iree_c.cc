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
#include <vector>

#include "android/cpp/c/backend_c.h"
#include "android/cpp/c/type.h"
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

  // Input buffers.
  int32_t input_count = 0;
  std::vector<iree_hal_buffer_view_t*> input_buffers;
  std::vector<mlperf_data_t> input_types;

  // Outputs.
  int32_t output_count = 0;
  std::vector<iree_hal_buffer_t*> output_buffers;
  std::vector<mlperf_data_t> output_types;

  int32_t num_queries = 0;
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

void cleanup(IreeBackendData* backend_data) {
  if (backend_data) {
    for (auto* buffer_view: backend_data->input_buffers) {
      iree_hal_buffer_view_release(buffer_view);
    }
    for (auto* buffer : backend_data->output_buffers) {
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
  for (int i = 0; i < configs->count; ++i) {
    if (strcmp(configs->keys[i], "driver") == 0) {
      driver = iree_make_cstring_view(configs->values[i]);
    } else if (strcmp(configs->keys[i], "entry_function") == 0) {
      entry_function = iree_make_cstring_view(configs->values[i]);
    } else if (strcmp(configs->keys[i], "function_inputs") == 0) {
      function_inputs = iree_make_cstring_view(configs->values[i]);
    } else if (strcmp(configs->keys[i], "function_outputs") == 0) {
      function_outputs = iree_make_cstring_view(configs->values[i]);
    }
  }

  IreeBackendData* backend_data = new IreeBackendData();
  backendExists = true;

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
  backend_data->output_count = (int32_t) results.size;
  backend_data->output_buffers.resize(backend_data->output_count);
  backend_data->output_types.resize(backend_data->output_count);

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
    for (int j = 0; j < shape_rank; ++j) {
      element_size *= shape[j];
    }

    type_str = iree_string_view_substr(output, last_x_index + 1,
                                       IREE_STRING_VIEW_NPOS);
    iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
    iree_hal_parse_element_type(type_str, &element_type);

    backend_data->output_types[i].type = IreeType2Type(element_type);
    backend_data->output_types[i].size = element_size;
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
  return backend_data->output_count;
}

// Return the type of ith output.
mlperf_data_t mlperf_backend_get_output_type(mlperf_backend_ptr_t backend_ptr,
                                             int32_t i) {
  IreeBackendData* backend_data = (IreeBackendData*) backend_ptr;
  return backend_data->output_types[i];
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
  iree_vm_list_t* output_list =
      iree_runtime_call_outputs(&backend_data->iree_runtime_call);
  iree_vm_ref_t value = {0};
  iree_status_t status = iree_vm_list_get_ref_assign(output_list, i, &value);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MLPERF_FAILURE;
  }

  iree_hal_buffer_view_t* buffer_view = nullptr;
  status = iree_hal_buffer_view_check_deref(value, &buffer_view);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MLPERF_FAILURE;
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
    return MLPERF_FAILURE;
  }

  if (backend_data->output_buffers[i]) {
    iree_hal_buffer_release(backend_data->output_buffers[i]);
    backend_data->output_buffers[i] = nullptr;
  }
  backend_data->output_buffers[i] = buffer;
  iree_hal_buffer_retain(backend_data->output_buffers[i]);
  *data = buffer_mapping.contents.data;
  return MLPERF_SUCCESS;
}
