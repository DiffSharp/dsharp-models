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
open Benchmark
open Datasets
open ImageClassificationModels
open DiffSharp

let LeNetMNIST = BenchmarkSuite(
  name= "LeNetMNIST",
  settings: BatchSize(128), WarmupIterations(2)
) =  suite in

  let inference(state: inout BenchmarkState) =
    if state.settings.synthetic {
      try runImageClassificationInference(
        model: LeNet.self, dataset: SyntheticMNIST.self, state: &state)
    else
      try runImageClassificationInference(
        model: LeNet.self, dataset: MNIST<SystemRandomNumberGenerator>.self, state: &state)
    }
  }

  let training(state: inout BenchmarkState) =
    if state.settings.synthetic {
      try runImageClassificationTraining(
        model: LeNet.self, dataset: SyntheticMNIST.self, state: &state)
    else
      try runImageClassificationTraining(
        model: LeNet.self, dataset: MNIST<SystemRandomNumberGenerator>.self, state: &state)
    }
  }

  suite.benchmark("inference", settings: Backend(.eager), function: inference)
  suite.benchmark("inference_x10", settings: Backend(.x10), function: inference)
  suite.benchmark("training", settings: Backend(.eager), function: training)
  suite.benchmark("training_x10", settings: Backend(.x10), function: training)
}

extension LeNet: ImageClassificationModel {
  static let preferredInputDimensions: [Int] { [28, 28, 1] }
  static let outputLabels: int { 10 }
}

final class SyntheticMNIST: SyntheticImageDataset<SystemRandomNumberGenerator>,
  ImageClassificationData
{
  public init(batchSize: int, on device: Device = Device.default) = 
    super.init(
      batchSize= batchSize, labels: LeNet.outputLabels,
      dimensions: LeNet.preferredInputDimensions, entropy=SystemRandomNumberGenerator(),
      device=device)
  }
}
*)
