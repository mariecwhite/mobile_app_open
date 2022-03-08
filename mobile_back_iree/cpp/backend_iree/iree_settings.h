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
  benchmark_id: "squad"
  accelerator: ""
  accelerator_desc: ""
  configuration: "Squad"
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

benchmark_setting {
  benchmark_id: "imagenet"
  accelerator: ""
  accelerator_desc: ""
  configuration: "Imagenet"
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
    value: "1x224x224x3xf32"
  }
  custom_setting {
    id: "function_outputs"
    value: "1x1001xf32"
  }
}

benchmark_setting {
  benchmark_id: "imagenet_quantized"
  accelerator: ""
  accelerator_desc: ""
  configuration: "Imagenet"
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
    value: "1x224x224x3xui8"
  }
  custom_setting {
    id: "function_outputs"
    value: "1x1001xui8"
  }
}

benchmark_setting {
  benchmark_id: "ade20k"
  accelerator: ""
  accelerator_desc: ""
  configuration: "ADE20K"
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
    value: "1x512x512x3xf32"
  }
  custom_setting {
    id: "function_outputs"
    value: "1x512x512xi32"
  }
}

benchmark_setting {
  benchmark_id: "ade20k_quantized"
  accelerator: ""
  accelerator_desc: ""
  configuration: "ADE20K"
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
    value: "1x512x512x3xui8"
  }
  custom_setting {
    id: "function_outputs"
    value: "1x512x512xi32"
  }
}

)SETTINGS";

#endif  // IREE_SETTINGS_H
