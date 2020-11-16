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

open Datasets
open ImageClassificationModels
open DiffSharp
open TrainingLoop

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

let dataset = Imagenette(batchSize= 64, inputSize= .resized320, outFeatures=224, device=device)
let model = MobileNetV2(classCount: 10)
let optimizer = SGD(model, learningRate=0.002, momentum: 0.9)

let trainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  optimizer=optimizer,
  lossFunction: softmaxCrossEntropy,
  metrics: [.accuracy])

try! trainingLoop.fit(&model, epochs: 10, device=device)
