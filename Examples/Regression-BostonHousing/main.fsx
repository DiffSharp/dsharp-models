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

open Datasets
open DiffSharp
open DiffSharp.Model
open DiffSharp.Data

// Defining default hyper parameters
let batchSize = 50
let numEpochs = 200
let learningRate = 0.0001
let sizeHidden1 = 100
let sizeHidden2 = 50
let sizeHidden3 = 10
let sizeHidden4 = 1


// open Dataset
let dataset = BostonHousing()
let trainIter = DataLoader(TensorDataset(dataset.xTrain,dataset.yTrain),batchSize=batchSize, shuffle = true)

// Create Model
type RegressionModel() = 
    inherit Model()
    let layer1 = Linear(inFeatures=13, outFeatures=sizeHidden1) --> dsharp.relu
    let layer2 = Linear(inFeatures=sizeHidden1, outFeatures=sizeHidden2) --> dsharp.relu
    let layer3 = Linear(inFeatures=sizeHidden2, outFeatures=sizeHidden3) --> dsharp.relu
    let layer4 = Linear(inFeatures=sizeHidden3,outFeatures=sizeHidden4)
    
    do base.add([layer1;layer2;layer3;layer4],
                ["layer1";"layer2";"layer3";"layer4"])
    override _.forward(input) =
        input |> layer1.forward |> layer2.forward |> layer3.forward |> layer4.forward

let model = RegressionModel()
model.mode <- Mode.Train

let criterion (outputs, labels) = dsharp.mseLoss(outputs,labels)
// Train Model
let optimizer = Optim.SGD(model, learningRate= dsharp.scalar learningRate)

for epoch in 1..numEpochs do
    let mutable runningLoss = 0.0
    for _batch, inputs, labels in trainIter.epoch() do
        model.reverseDiff()
        // forward pass
        let outputs = model.forward(inputs)
        // defining loss
        let loss = criterion(outputs,labels)
        // computing gradients
        loss.reverse()
        // accumulating running loss
        runningLoss <- runningLoss + loss.toDouble()
        // updated weights based on computed gradients
        optimizer.step()
    if epoch % 20 = 0 then
        printfn $"Epoch {epoch}/{numEpochs} running accumulative loss across all batches: %.3f{runningLoss}"

let outputs = model.forward(dataset.xTest)
let err = sqrt(dsharp.mseLoss(outputs,dataset.yTest))