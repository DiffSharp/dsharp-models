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

#r @"..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\bin\Debug\netcoreapp3.1\publish\Library.dll"

open Datasets
open DiffSharp
open DiffSharp.Model

// open Dataset
let dataset = BostonHousing()

// Create Model
type RegressionModel() = 
    inherit Model()
    let layer1 = Linear(inFeatures=13, outFeatures=64, activation=dsharp.relu)
    let layer2 = Linear(inFeatures=64, outFeatures=32, activation=dsharp.relu)
    let layer3 = Linear(inFeatures=32, outFeatures=1)
    
    override _.forward(input) =
        input |> layer1.forward |> layer2.forward |> layer3.forward

let model = RegressionModel()

// Train Model
let optimizer = RMSProp(model, learningRate=0.001)
model.mode <- Mode.Train

let epochCount = 500
let batchSize = 32
let numberOfBatch = int(ceil(Double(dataset.numTrainRecords) / double(batchSize)))
let shuffle = true

let meanAbsoluteError(predictions=Tensor, truths: Tensor) =
    abs(Tensor(predictions - truths)).mean().toScalar()


print("Starting training..")

for epoch in 1..epochCount do
    let epochLoss: double = 0
    let epochMAE: double = 0
    let batchCount: int = 0
    let batchArray = Array.replicate false, count: numberOfBatch)
    for batch in 0..numberOfBatch-1 do
        let r = batch
        if shuffle then
            while true do
                r = Int.random(0..numberOfBatch-1)
                if not batchArray.[r] then
                    batchArray.[r] = true
                    break

        let batchStart = r * batchSize
        let batchEnd = min(dataset.numTrainRecords, batchStart + batchSize)
        let (loss, grad) = valueWithGradient<| fun model -> =  (model: RegressionModel) = Tensor in
            let logits = model(dataset.xTrain[batchStart..<batchEnd])
            meanSquaredError(predicted=logits, expected=dataset.yTrain[batchStart..<batchEnd])

        optimizer.update(&model, along=grad)
        
        let logits = model(dataset.xTrain[batchStart..<batchEnd])
        epochMAE <- epochMAE + meanAbsoluteError(predictions=logits, truths: dataset.yTrain[batchStart..<batchEnd])
        epochLoss <- epochLoss + loss.toScalar()
        batchCount <- batchCount + 1

    epochMAE /= double(batchCount)
    epochLoss /= double(batchCount)

    if epoch = epochCount-1 then
        print($"MSE: {epochLoss}, MAE: {epochMAE}, Epoch: \(epoch+1)")



// Evaluate Model

print("Evaluating model..")

model.mode <- Mode.Eval

let prediction = model(dataset.xTest)

let evalMse = meanSquaredError(predicted=prediction, expected=dataset.yTest).toScalar()/double(dataset.numTestRecords)
let evalMae = meanAbsoluteError(predictions=prediction, truths: dataset.yTest)/double(dataset.numTestRecords)

print($"MSE: {evalMse}, MAE: {evalMae}")
