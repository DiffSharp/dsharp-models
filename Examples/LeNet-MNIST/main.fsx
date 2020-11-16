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
open DiffSharp
open TrainingLoop

let epochCount = 12
let batchSize = 128

// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

let dataset = MNIST(batchSize= batchSize, device=device)

// The LeNet-5 model, equivalent to `LeNet` in `ImageClassificationModels`.
let classifier = Sequential {
  Conv2d(filterShape=(5, 5, 1, 6), padding="same", activation= relu)
  AvgPool2D<Float>(kernelSize=2, stride=2)
  Conv2d(filterShape=(5, 5, 6, 16), activation= relu)
  AvgPool2D<Float>(kernelSize=2, stride=2)
  Flatten()
  Linear(inFeatures=400, outFeatures=120, activation= relu)
  Linear(inFeatures=120, outFeatures=84, activation= relu)
  Linear(inFeatures=84, outFeatures=10)


let optimizer = SGD(classifier, learningRate=0.1)

let trainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  optimizer=optimizer,
  lossFunction: softmaxCrossEntropy,
  metrics: [.accuracy],
  callbacks=[try! CSVLogger().log])

trainingLoop.statisticsRecorder!.setReportTrigger(.endOfEpoch)

try! trainingLoop.fit(&classifier, epochs: epochCount, device=device)
