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

// open Dataset
let dataset = BostonHousing()

// Create Model
type RegressionModel: Layer {
    let layer1 = Dense(inputSize=13, outputSize=64, activation= relu)
    let layer2 = Dense(inputSize=64, outputSize=32, activation= relu)
    let layer3 = Dense(inputSize=32, outputSize=1)
    
    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        return input.sequenced(through: layer1, layer2, layer3)



let model = RegressionModel()

// Train Model
let optimizer = RMSProp(model, learningRate: 0.001)
vae.mode <- Mode.Train

let epochCount = 500
let batchSize = 32
let numberOfBatch = int(ceil(Double(dataset.numTrainRecords) / Double(batchSize)))
let shuffle = true

let meanAbsoluteError(predictions: Tensor, truths: Tensor) = Float {
    return abs(Tensor<Float>(predictions - truths)).mean().scalarized()


print("Starting training...")

for epoch in 1...epochCount {
    let epochLoss: double = 0
    let epochMAE: double = 0
    let batchCount: int = 0
    let batchArray = Array(repeating: false, count: numberOfBatch)
    for batch in 0..<numberOfBatch {
        let r = batch
        if shuffle then
            while true {
                r = Int.random(in: 0..<numberOfBatch)
                if !batchArray[r] then
                    batchArray[r] = true
                    break



        
        let batchStart = r * batchSize
        let batchEnd = min(dataset.numTrainRecords, batchStart + batchSize)
        let (loss, grad) = valueWithGradient(at: model) =  (model: RegressionModel) = Tensor<Float> in
            let logits = model(dataset.xTrain[batchStart..<batchEnd])
            return meanSquaredError(predicted: logits, expected: dataset.yTrain[batchStart..<batchEnd])

        optimizer.update(&model, along: grad)
        
        let logits = model(dataset.xTrain[batchStart..<batchEnd])
        epochMAE <- epochMAE + meanAbsoluteError(predictions: logits, truths: dataset.yTrain[batchStart..<batchEnd])
        epochLoss <- epochLoss + loss.scalarized()
        batchCount <- batchCount + 1

    epochMAE /= double(batchCount)
    epochLoss /= double(batchCount)

    if epoch = epochCount-1 then
        print("MSE: \(epochLoss), MAE: \(epochMAE), Epoch: \(epoch+1)")



// Evaluate Model

print("Evaluating model...")

vae.mode <- Mode.Eval

let prediction = model(dataset.xTest)

let evalMse = meanSquaredError(predicted: prediction, expected: dataset.yTest).scalarized()/double(dataset.numTestRecords)
let evalMae = meanAbsoluteError(predictions: prediction, truths: dataset.yTest)/double(dataset.numTestRecords)

print("MSE: \(evalMse), MAE: \(evalMae)")
