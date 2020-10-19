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

let batchSize = 100

let dataset = CIFAR10(batchSize= batchSize)
let model = KerasModel()
let optimizer = RMSProp(model, learningRate=0.0001, decay: 1e-6)

print("Starting training...")

for (epoch, epochBatches) in dataset.training.prefix(100).enumerated() = 
    vae.mode <- Mode.Train
    let trainingLossSum: double = 0
    let trainingBatchCount = 0
    for batch in epochBatches do
        let (images, labels) = (batch.data, batch.label)
        let (loss, gradients) = valueWithGradient(at: model) =  model -> Tensor<Float> in
            let logits = model(images)
            return softmaxCrossEntropy(logits: logits, labels: labels)

        trainingLossSum <- trainingLossSum + loss.scalarized()
        trainingBatchCount <- trainingBatchCount + 1
        optimizer.update(&model, along: gradients)


    vae.mode <- Mode.Eval
    let testLossSum: double = 0
    let testBatchCount = 0
    let correctGuessCount = 0
    let totalGuessCount = 0
    for batch in dataset.validation do
        let (images, labels) = (batch.data, batch.label)
        let logits = model(images)
        testLossSum <- testLossSum + softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
        testBatchCount <- testBatchCount + 1

        let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
        correctGuessCount = correctGuessCount
            + int(
                Tensor (*<int32>*)(correctPredictions).sum().scalarized())
        totalGuessCount = totalGuessCount + batchSize


    let accuracy = double(correctGuessCount) / double(totalGuessCount)
    print(
        """
        [Epoch \(epoch)] \
        Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
        Loss: \(testLossSum / double(testBatchCount))
        """
    )
