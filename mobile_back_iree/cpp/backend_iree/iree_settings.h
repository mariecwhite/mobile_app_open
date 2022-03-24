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
  custom_setting {
    id: "function_output_scales"
    value: ""
  }
  custom_setting {
    id: "function_output_zero_points"
    value: ""
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
  custom_setting {
    id: "function_output_scales"
    value: ""
  }
  custom_setting {
    id: "function_output_zero_points"
    value: ""
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
  custom_setting {
    id: "function_output_scales"
    value: ""
  }
  custom_setting {
    id: "function_output_zero_points"
    value: ""
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
  custom_setting {
    id: "function_output_scales"
    value: ""
  }
  custom_setting {
    id: "function_output_zero_points"
    value: ""
  }
}

benchmark_setting {
  benchmark_id: "coco"
  accelerator: ""
  accelerator_desc: ""
  configuration: "COCO"
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
    value: "1x320x320x3xf32"
  }
  custom_setting {
    id: "function_outputs"
    value: "1x2034x4xf32,1x2034x91xf32"
  }
  custom_setting {
    id: "function_output_scales"
    value: ""
  }
  custom_setting {
    id: "function_output_zero_points"
    value: ""
  }
}

benchmark_setting {
  benchmark_id: "coco_ssd_mobilenet_v1_quantized"
  accelerator: ""
  accelerator_desc: ""
  configuration: "COCO"
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
    value: "1x320x320x3xui8"
  }
  custom_setting {
    id: "function_outputs"
    value: "1x20x20x12xui8,1x20x20x273xui8,1x10x10x24xui8,1x10x10x546xui8,1x5x5x24xui8,1x5x5x546xui8,1x3x3x24xui8,1x3x3x546xui8,1x2x2x24xui8,1x2x2x546xui8,1x1x1x24xui8,1x1x1x546xui8"
  }
  custom_setting {
    id: "function_output_scales"
    value: "0.0873504,0.0485853,0.0343766,0.0474807,0.0291031,0.0406678,0.0273419,0.0389664,0.0201044,0.0293194,0.0198375,0.0310718"
  }
  custom_setting {
    id: "function_output_zero_points"
    value: "181,232,141,220,126,217,123,219,138,231,150,220"
  }
}

benchmark_setting {
  benchmark_id: "coco_spaghettinet_edgetpu_l_uint8"
  accelerator: ""
  accelerator_desc: ""
  configuration: "COCO"
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
    value: "1x320x320x3xui8"
  }
  custom_setting {
    id: "function_outputs"
    value: "1x20x20x12xui8,1x20x20x273xui8,1x10x10x24xui8,1x10x10x546xui8,1x5x5x24xui8,1x5x5x546xui8,1x3x3x24xui8,1x3x3x546xui8,1x3x3x24xui8,1x3x3x546xui8"
  }
  custom_setting {
    id: "function_output_scales"
    value: "0.0907745,0.0482535,0.0341228,0.0453906,0.0305526,0.0411499,0.0245913,0.0380795,0.0233667,0.0362647"
  }
  custom_setting {
    id: "function_output_zero_points"
    value: "181,227,141,213,121,217,125,218,134,222"
  }
}

)SETTINGS";

#endif  // IREE_SETTINGS_H
