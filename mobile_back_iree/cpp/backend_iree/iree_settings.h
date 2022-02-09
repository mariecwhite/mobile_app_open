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
#include <string>

#ifndef IREE_SETTINGS_H
#define IREE_SETTINGS_H

const std::string iree_settings = R"SETTINGS(
benchmark_setting {
  benchmark_id: "mobilebert"
  accelerator: ""
  accelerator_desc: ""
  configuration: "MobileBert"
  src: ""
  custom_setting {
    id: "driver"
    value: "dylib"
  }
  custom_setting {
    id: "entry_function"
    value: "module.main"
  }
  custom_setting {
    id: "function_inputs"
    value: "1x384xi32,1x384xi32,1x384xi32"
  }
  custom_setting {
    id: "function_outputs"
    value: "1x384xf32,1x384xf32"
  }
}
)SETTINGS";

#endif  // IREE_SETTINGS_H
