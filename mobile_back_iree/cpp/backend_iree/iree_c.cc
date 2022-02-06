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
  iree_hal_dim_t input_dims[2] = {1, 384};
  std::vector<int32_t> input_word_ids = std::vector<int32_t>(384);
  std::vector<int32_t> input_type_ids = std::vector<int32_t>(384);
  std::vector<int32_t> input_mask = std::vector<int32_t>(384);

  // Outputs.
  int32_t output_count = 0;
  iree_hal_buffer_t* output_buffers[2] = {nullptr, nullptr};
  iree_hal_buffer_mapping_t output_buffer_mappings[2];
};

static bool backendExists = false;

void cleanup(IreeBackendData* backend_data) {
  printf("Calling cleanup");
  if (backend_data) {
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
  printf("IREE runtime matches hardware");
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
    printf("Error: Failed to create IREE runtime instance.");
    cleanup(backend_data);
    return nullptr;
  }

  // Create IREE runtime session.
  iree_hal_device_t* device = nullptr;
  status = iree_runtime_instance_try_create_default_device(
      backend_data->iree_runtime_instance, iree_make_cstring_view("dylib"),
      &device);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to create default device.");
    cleanup(backend_data);
    return nullptr;
  }

  iree_runtime_session_options_initialize(&backend_data->session_options);
  status = iree_runtime_session_create_with_device(
      backend_data->iree_runtime_instance, &backend_data->session_options, device,
      iree_runtime_instance_host_allocator(backend_data->iree_runtime_instance),
      &backend_data->iree_runtime_session);
  iree_hal_device_release(device);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to create IREE runtime session.");
    cleanup(backend_data);
    return nullptr;
  }

  // Load module.
  status = iree_runtime_session_append_bytecode_module_from_file(backend_data->iree_runtime_session, "/tmp/mb_float.vmfb");
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to load IREE module.");
    cleanup(backend_data);
    return nullptr;
  }

  // Initialize the call function.
  status = iree_runtime_call_initialize_by_name(
      backend_data->iree_runtime_session, iree_make_cstring_view("module.main"),
      &backend_data->iree_runtime_call);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to load IREE module.");
    cleanup(backend_data);
    return nullptr;
  }

  // Get input/output metadata.
  iree_vm_function_t main;
  status = iree_runtime_session_lookup_function(
      backend_data->iree_runtime_session, iree_make_cstring_view("module.main"),
      &main);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to lookup function.");
    cleanup(backend_data);
    return nullptr;
  }
  iree_vm_function_signature_t main_signature = iree_vm_function_signature(&main);
  iree_string_view_t arguments, results;
  status = iree_vm_function_call_get_cconv_fragments(
      &main_signature, &arguments, &results);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed iree_vm_function_call_get_cconv_fragments");
    cleanup(backend_data);
    return nullptr;
  }
  backend_data->input_count = (int32_t)arguments.size;
  backend_data->output_count = (int32_t)results.size;
  printf("Input count: %d, Output count: %d\n", backend_data->input_count,
         backend_data->output_count);

  // Append inputs to the allocators.
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(backend_data->iree_runtime_session);

  // %arg0: tensor<1x384xi32>
  iree_hal_buffer_view_t* arg0 = nullptr;
  status = iree_hal_buffer_view_wrap_or_clone_heap_buffer(
      device_allocator, backend_data->input_dims,
      IREE_ARRAYSIZE(backend_data->input_dims), IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_byte_span((void*)backend_data->input_word_ids.data(),
                          backend_data->input_word_ids.size() *
                              sizeof(backend_data->input_word_ids[0])),
      iree_allocator_null(), &arg0);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to create buffer view for input arg0.");
    cleanup(backend_data);
    return nullptr;
  }
  status = iree_runtime_call_inputs_push_back_buffer_view(
      &backend_data->iree_runtime_call, arg0);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to append input arg0.");
    cleanup(backend_data);
    return nullptr;
  }

  iree_hal_buffer_view_release(arg0);

  // %arg1: tensor<1x384xi32>
  iree_hal_buffer_view_t* arg1 = nullptr;
  status = iree_hal_buffer_view_wrap_or_clone_heap_buffer(
      device_allocator, backend_data->input_dims,
      IREE_ARRAYSIZE(backend_data->input_dims), IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_byte_span((void*)backend_data->input_type_ids.data(),
                          backend_data->input_type_ids.size() *
                              sizeof(backend_data->input_type_ids[0])),
      iree_allocator_null(), &arg1);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to create buffer view for input arg1.");
    cleanup(backend_data);
    return nullptr;
  }
  status = iree_runtime_call_inputs_push_back_buffer_view(
      &backend_data->iree_runtime_call, arg1);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to append input arg1.");
    cleanup(backend_data);
    return nullptr;
  }
  iree_hal_buffer_view_release(arg1);

  // %arg2: tensor<1x384xi32>
  iree_hal_buffer_view_t* arg2 = nullptr;
  status = iree_hal_buffer_view_wrap_or_clone_heap_buffer(
      device_allocator, backend_data->input_dims,
      IREE_ARRAYSIZE(backend_data->input_dims), IREE_HAL_ELEMENT_TYPE_INT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_ALL,
      iree_make_byte_span((void*)backend_data->input_mask.data(),
                          backend_data->input_mask.size() *
                              sizeof(backend_data->input_mask[0])),
      iree_allocator_null(), &arg2);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to create buffer view for input arg2.");
    cleanup(backend_data);
    return nullptr;
  }
  status = iree_runtime_call_inputs_push_back_buffer_view(
      &backend_data->iree_runtime_call, arg2);
  if (!iree_status_is_ok(status)) {
    printf("Error: Failed to append input arg2.");
    cleanup(backend_data);
    return nullptr;
  }
  iree_hal_buffer_view_release(arg2);

  printf("Backend successfully created.\n");
  return backend_data;
}

// Vendor name who create this backend.
const char* mlperf_backend_vendor_name(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*)backend_ptr;
  return backend_data->vendor;
}

// Return the name of this backend.
const char* mlperf_backend_name(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*)backend_ptr;
  return backend_data->name;
}

// Destroy the backend pointer and its data.
void mlperf_backend_delete(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*)backend_ptr;
  cleanup(backend_data);
}

// Run the inference for a sample.
mlperf_status_t mlperf_backend_issue_query(mlperf_backend_ptr_t backend_ptr) {
  printf("Issuing query\n");
  IreeBackendData* backend_data = (IreeBackendData*)backend_ptr;

  iree_status_t status =
      iree_runtime_call_invoke(&backend_data->iree_runtime_call, /*flags=*/0);
  if (!iree_status_is_ok(status)) {
    printf("Error returned when calling invoke");
    iree_status_fprint(stderr, status);
    return MLPERF_FAILURE;
  }
  printf("Query issued\n");
  return MLPERF_SUCCESS;
}

// Flush the staged queries immediately.
mlperf_status_t mlperf_backend_flush_queries(mlperf_backend_ptr_t backend_ptr) {
  return MLPERF_SUCCESS;
}

// Return the number of inputs of the model.
int32_t mlperf_backend_get_input_count(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*)backend_ptr;
  printf("Returning input count: %d", backend_data->input_count);
  return backend_data->input_count;
}

// Return the type of the ith input.
mlperf_data_t mlperf_backend_get_input_type(mlperf_backend_ptr_t backend_ptr,
                                            int32_t i) {
  mlperf_data_t type;
  type.type = mlperf_data_t::Int32;
  type.size = 384;
  return type;
}

// Set the data for ith input.
mlperf_status_t mlperf_backend_set_input(mlperf_backend_ptr_t backend_ptr,
                                         int32_t batch_index, int32_t i,
                                         void* data) {
  IreeBackendData* backend_data = (IreeBackendData*)backend_ptr;
  if (i == 0) {
    memcpy(&backend_data->input_word_ids[0], data,
           backend_data->input_word_ids.size() *
               sizeof(backend_data->input_word_ids[0]));
  } else if (i == 1) {
    memcpy(&backend_data->input_type_ids[0], data,
           backend_data->input_type_ids.size() *
               sizeof(backend_data->input_type_ids[0]));
  } else if (i == 2) {
    memcpy(
        &backend_data->input_mask[0], data,
        backend_data->input_mask.size() * sizeof(backend_data->input_mask[0]));
  } else {
    printf("Invalid input index");
    mlperf_backend_delete(backend_data);
    return MLPERF_FAILURE;
  }
  printf("Input set for batch_index %d, input index %d\n", batch_index, i);
  return MLPERF_SUCCESS;
}

// Return the number of outputs for the model.
int32_t mlperf_backend_get_output_count(mlperf_backend_ptr_t backend_ptr) {
  IreeBackendData* backend_data = (IreeBackendData*)backend_ptr;
  printf("Returning output count: %d", backend_data->output_count);
  return backend_data->output_count;
}

// Return the type of ith output.
mlperf_data_t mlperf_backend_get_output_type(mlperf_backend_ptr_t backend_ptr,
                                             int32_t i) {
  mlperf_data_t type;
  type.type = mlperf_data_t::Float32;
  type.size = 384;
  return type;
}

// Get the data from ith output.
mlperf_status_t mlperf_backend_get_output(mlperf_backend_ptr_t backend_ptr,
                                          uint32_t batch_index, int32_t i,
                                          void** data) {
  if (batch_index > 0) {
    printf("Only batch size 1 is currently supported.");
    return MLPERF_FAILURE;
  }

  IreeBackendData* backend_data = (IreeBackendData*)backend_ptr;
  printf("Retrieving output for batch %d, index %d\n", batch_index, i);

  iree_vm_list_t* output_list = iree_runtime_call_outputs(&backend_data->iree_runtime_call);

  iree_vm_ref_t value = {0};
  iree_status_t status = iree_vm_list_get_ref_assign(output_list, i, &value);
  if (!iree_status_is_ok(status)) {
    printf("iree_vm_list_get_ref_assign failed");
    iree_status_fprint(stderr, status);
    return MLPERF_FAILURE;
  }

  iree_hal_buffer_view_t* buffer_view = nullptr;
  status = iree_hal_buffer_view_check_deref(value, &buffer_view);
  if (!iree_status_is_ok(status)) {
    printf("iree_hal_buffer_view_check_deref failed");
    iree_status_fprint(stderr, status);
    return MLPERF_FAILURE;
  }

  iree_device_size_t byte_offset = 0;
  iree_device_size_t byte_length = IREE_WHOLE_BUFFER;
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
  status = iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, byte_offset,
      byte_length, &backend_data->output_buffer_mappings[i]);
  if (!iree_status_is_ok(status)) {
    printf("Error mapping buffer.");
    iree_status_fprint(stderr, status);
    return MLPERF_FAILURE;
  }

  backend_data->output_buffers[i] = buffer;
  iree_hal_buffer_retain(backend_data->output_buffers[i]);
  *data = backend_data->output_buffer_mappings[i].contents.data;
  printf("Output contents: %c\n", static_cast<char*>(*data)[0]);
  return MLPERF_SUCCESS;
}
