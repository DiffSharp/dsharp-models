// Copyright 2019 The TensorFlow Authors, adapted by the DiffSharp authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

namespace Benchmark
(*
open ImageClassificationModels

let ResNetSuites = [
  //
  // Cifar input dimensions. 
  //
  makeLayerSuite(
    name= "ResNet18",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) = 
    ResNet(classCount: 10, depth: ResNet18, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name= "ResNet34",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) = 
    ResNet(classCount: 10, depth: ResNet34, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name= "ResNet50",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) = 
    ResNet(classCount: 10, depth: ResNet50, downsamplingInFirstStage: true, useLaterStride: false)
  },
  makeLayerSuite(
    name= "ResNet56",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) = 
    ResNet(classCount: 10, depth: ResNet56, downsamplingInFirstStage: true, useLaterStride: false)
  },
  makeLayerSuite(
    name= "ResNet101",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) = 
    ResNet(classCount: 10, depth: ResNet101, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name= "ResNet152",
    inputDimensions: cifarInput,
    outputDimensions: cifarOutput
  ) = 
    ResNet(classCount: 10, depth: ResNet152, downsamplingInFirstStage: true)
  },
  //
  // ImageNet dimensions.
  //
  makeLayerSuite(
    name= "ResNet18",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNet(classCount: 1000, depth: ResNet18)
  },
  makeLayerSuite(
    name= "ResNet34",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNet(classCount: 1000, depth: ResNet34)
  },
  makeLayerSuite(
    name= "ResNet50",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNet(classCount: 1000, depth: ResNet50, useLaterStride: false)
  },
  makeLayerSuite(
    name= "ResNet56",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNet(classCount: 1000, depth: ResNet56)
  },
  makeLayerSuite(
    name= "ResNet101",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNet(classCount: 1000, depth: ResNet101)
  },
  makeLayerSuite(
    name= "ResNet152",
    inputDimensions: imageNetInput,
    outputDimensions: imageNetOutput
  ) = 
    ResNet(classCount: 1000, depth: ResNet152)
  },
]
*)
