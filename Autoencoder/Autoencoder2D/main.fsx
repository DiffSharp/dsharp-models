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

/// Based on https://blog.keras.io/building-autoencoders-in-keras.html
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r @"System.Runtime.Extensions.dll"

open Datasets
open DiffSharp
open DiffSharp.Util
open DiffSharp.Model

let epochCount = 10
let batchSize = 100
let imageHeight = 28
let imageWidth = 28

let outputFolder = "./output/"
let dataset = KuzushijiMNIST(batchSize=batchSize, flattening=true)

// An autoencoder.
type Autoencoder2D() =
    inherit Model()
    let encoder1 = Conv2d(1, 16, kernelSize=3, padding=1)
    let encoder2 = Function (fun t -> t.maxpool2d(kernelSize=2, stride=2, padding=1))
    let encoder3 = Conv2d(16, 8, kernelSize=3, padding=1)
    let encoder4 = Function (fun t -> t.maxpool2d(kernelSize=2, stride=2, padding=1))
    let encoder5 = Conv2d(8, 8, kernelSize=3, padding=1)
    let encoder6 = Function (fun t -> t.maxpool2d(kernelSize=2, stride=2, padding=1))

    let decoder1 = Conv2d(8, 8, kernelSize=3, padding=3/2 (* "same" *) )
    let decoder2 = UpSampling2d(size=2)
    let decoder3 = Conv2d(8, 8, kernelSize=3, padding=3/2 (* "same" *) )
    let decoder4 = UpSampling2d(size=2)
    let decoder5 = Conv2d(8, 16, kernelSize=3)
    let decoder6 = UpSampling2d(size=2)

    let output = Conv2d(16, 1, kernelSize=3, padding=3/2, activation=dsharp.sigmoid)
     
    override _.forward(input) =
        let resize = input.view([batchSize; 28; 28; 1])
        let encoder =
            resize 
            |> encoder1.forward |> dsharp.relu
            |> encoder2.forward 
            |> encoder3.forward |> dsharp.relu
            |> encoder4.forward 
            |> encoder5.forward |> dsharp.relu
            |> encoder6.forward 
        let decoder = encoder |> decoder1.forward |> decoder2.forward |> decoder3.forward |> decoder4.forward |> decoder5.forward |> decoder6.forward
        output.forward(decoder).view([batchSize; imageHeight * imageWidth])



let model = Autoencoder2D()
let optimizer = AdaDelta(model)

// Training loop
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() do
    model.mode <- Mode.Train
    for batch in epochBatches do
        let x = batch.data

        let image, δmodel = model.gradv(image, fun x -> meanSquaredError(predicted=image, expected=x))

        optimizer.update(&model, along=δmodel)


    model.mode <- Mode.Eval
    let testLossSum: double = 0
    let testBatchCount = 0
    for batch in dataset.validation do
        let sampleImages = batch.data
        let testImages = model(sampleImages)

        try
            dsharp.saveImage(
                sampleImages.[0..0], shape=[imageWidth; imageHeight], format="grayscale",
                directory=outputFolder, name= $"epoch-{epoch}-input")
            dsharp.saveImage(
                testImages.[0..0], shape=[imageWidth; imageHeight], format="grayscale",
                directory=outputFolder, name= $"epoch-{epoch}-output")
        with e ->
            print($"Could not save image with error: {error}")

        testLossSum <- testLossSum + meanSquaredError(predicted=testImages, expected=sampleImages).toScalar()
        testBatchCount <- testBatchCount + 1


    print($"""
        [Epoch {epoch}] \
        Loss: \(testLossSum / double(testBatchCount))
        """
    )

