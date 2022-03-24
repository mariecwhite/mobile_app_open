# Copyright 2022 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "center_size_encoded_boxes_proto",
    srcs = [
        "center_size_encoded_boxes.proto",
    ],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "center_size_encoded_boxes_cc_proto",
    deps = [":center_size_encoded_boxes_proto"],
)

cc_library(
    name = "detection_post_process",
    srcs = ["detection_post_process.cc"],
    hdrs = [
        "detection_post_process.h",
    ],
    deps = [
        ":center_size_encoded_boxes_cc_proto",
    ],
)
