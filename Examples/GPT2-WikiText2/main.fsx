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


#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r @"System.Runtime.Extensions.dll"

open Datasets
open DiffSharp
open TextModels
open TrainingLoop

// Avoid the eager mode runtime from taking all memory 
// and leaving none to X10 when run on the GPU.
_ = _ExecutionContext.global
// Until https://github.com/tensorflow/swift-apis/issues/993 is fixed, default to the eager-mode
// device on macOS instead of X10.
#if os(macOS)
  let device = Device.defaultTFEager
#else
  let device = Device.defaultXLA
#endif

let gpt = try GPT2()

let sequenceLength = gpt.contextSize
let trainingBatchSize = 2
let validationBatchSize = 2
let dataset = TextUnsupervised(bpe: gpt.bpe, variant=".wikiText2",
    trainingbatchSize= trainingBatchSize, validationbatchSize= validationBatchSize,
    sequenceLength=sequenceLength, device=device)
print("Dataset acquired.")

/// Reshape the `logits` and `labels` to required shape before calling
/// standard softmaxCrossEntropy API.
///
/// - Note: This can potentially be added to standard softmaxCrossEntropy API.

let softmaxCrossEntropyReshaped<Scalar>(logits:Tensor, labels=Tensor (*<int32>*)) = Tensor<
  Scalar
> where Scalar: TensorFlowFloatingPoint {
  return softmaxCrossEntropy(
  	logits=logits.view([logits.shape.dropLast().reduce(1, *), logits.shape |> Array.last]), 
  	labels=labels.view([labels.shape.reduce(1, *)]), 
  	reduction=_mean)


let trainingLoop: TrainingLoop = TrainingLoop(
  training: dataset.training,
  validation: dataset.validation,
  optimizer: Adam(gpt.model, learningRate=0.001),
  lossFunction: softmaxCrossEntropyReshaped,
  metrics: [.accuracy])

print("Starting training..")
try! trainingLoop.fit(&gpt.model, epochs: 10, device=device)
