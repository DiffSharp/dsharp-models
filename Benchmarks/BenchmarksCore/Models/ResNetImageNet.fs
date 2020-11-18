// Copyright 2020 The TensorFlow Authors, adapted by the DiffSharp authors. All Rights Reserved.
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
open Benchmark
open Datasets
open ImageClassificationModels
open DiffSharp

let ResNetImageNet = BenchmarkSuite(
  name= "ResNetImageNet",
  setDiffSha:pchSize(128), WarmupIterations(2), Synthetic(true)
) =  suite in

  let inference(state: inout BenchmarkState) =
    if state.settings.synthetic {
      try runImageClassificationInference(
        model: ResNet50.self, dataset: SyntheticImageNet.self, state: &state)
    else
      fatalError("Only synthetic ImageNet benchmarks are supported at the moment.")
    }
  }

  let training(state: inout BenchmarkState) =
    if state.settings.synthetic {
      try runImageClassificationTraining(
        model: ResNet50.self, dataset: SyntheticImageNet.self, state: &state)
    else
      fatalError("Only synthetic ImageNet benchmarks are supported at the moment.")
    }
  }

  suite.benchmark("inference", settings: Backend(.eager), function: inference)
  suite.benchmark("inference_x10", settings: Backend(.x10), function: inference)
  suite.benchmark("training", settings: Backend(.eager), function: training)
  suite.benchmark("training_x10", settings: Backend(.x10), function: training)
}

type ResNet50() = 
  inherit Model()
  let model: ResNet

  init() = 
    model = ResNet(classCount: 1000, depth: ResNet50, downsamplingInFirstStage: true)
  }

  
  let callAsFunction(input: Tensor<Float>) = Tensor<Float> =
    model(input)
  }
}

extension ResNet50: ImageClassificationModel {
  static let preferredInputDimensions: int[] { [224, 224, 3] }
  static let outputLabels: int { 1000 }
}

final class SyntheticImageNet: SyntheticImageDataset<SystemRandomNumberGenerator>,
  ImageClassificationData
{
  public init(batchSize: int, on device: Device = Device.default) = 
    super.init(
      batchSize= batchSize, labels=ResNet50.outputLabels,
      dimensions: ResNet50.preferredInputDimensions, entropy=SystemRandomNumberGenerator(),
      device=device)
  }
}
*)
