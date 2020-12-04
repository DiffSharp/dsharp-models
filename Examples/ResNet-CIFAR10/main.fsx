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

#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r @"System.Runtime.Extensions.dll"

open DiffSharp
open Datasets
open Models.ImageClassification
open DiffSharp
open TrainingLoop

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

let dataset = CIFAR10(batchSize= 10, device=device)
let model = ResNet(classCount=10, depth: ResNet56, downsamplingInFirstStage: false)
let optimizer = SGD(model, learningRate=dsharp.scalar 0.001)

let trainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  optimizer=optimizer,
  lossFunction: softmaxCrossEntropy,
  metrics: [.accuracy])

try! trainingLoop.fit(&model, epochs: 10, device=device)
