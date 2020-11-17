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
#load "Models.fsx"

open Datasets
open DiffSharp
open DiffSharp.Model
open DiffSharp.Util
open Models

let batchSize = 100

let dataset = CIFAR10(batchSize= batchSize)
let model = KerasModel()
let optimizer = RMSProp(model, learningRate=0.0001, decay=1e-6)

print("Starting training..")

for (epoch, epochBatches) in dataset.training.prefix(100).enumerated() do
    model.mode <- Mode.Train
    let mutable trainingLossSum: double = 0.0
    let mutable trainingBatchCount = 0
    for batch in epochBatches do
        let (images, labels) = (batch.data, batch.label)
        let (loss, gradients) = 
           valueWithGradient<| fun model -> 
            let logits = model(images)
            softmaxCrossEntropy(logits=logits, labels=labels)

        trainingLossSum <- trainingLossSum + loss.toScalar()
        trainingBatchCount <- trainingBatchCount + 1
        optimizer.update(&model, along=gradients)

    model.mode <- Mode.Eval
    let mutable testLossSum: double = 0.0
    let mutable testBatchCount = 0
    let mutable correctGuessCount = 0
    let mutable totalGuessCount = 0
    for batch in dataset.validation do
        let (images, labels) = (batch.data, batch.label)
        let logits = model.forward(images)
        testLossSum <- testLossSum + softmaxCrossEntropy(logits=logits, labels=labels).toScalar()
        testBatchCount <- testBatchCount + 1

        let correctPredictions = logits.argmax(dim=1) .== labels
        correctGuessCount <- correctGuessCount + int(dsharp.tensor(correctPredictions).sum().toScalar())
        totalGuessCount <- totalGuessCount + batchSize

    let accuracy = double(correctGuessCount) / double(totalGuessCount)
    print($"""
        [Epoch {epoch}] \
        Accuracy: {correctGuessCount}/{totalGuessCount} ({accuracy}) \
        Loss: {testLossSum / double(testBatchCount)}
        """
    )
