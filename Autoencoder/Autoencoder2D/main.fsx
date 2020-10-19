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

open Datasets


open DiffSharp

let epochCount = 10
let batchSize = 100
let imageHeight = 28
let imageWidth = 28

let outputFolder = "./output/"
let dataset = KuzushijiMNIST(batchSize= batchSize,  
    flattening=true)

// An autoencoder.
type Autoencoder2D: Layer {
    let encoder1 = Conv2d(filterShape=(3, 3, 1, 16), padding="same", activation= relu)
    let encoder2 = MaxPool2D<Float>(poolSize: (2, 2), stride=2, padding="same")
    let encoder3 = Conv2d(filterShape=(3, 3, 16, 8), padding="same", activation= relu)
    let encoder4 = MaxPool2D<Float>(poolSize: (2, 2), stride=2, padding="same")
    let encoder5 = Conv2d(filterShape=(3, 3, 8, 8), padding="same", activation= relu)
    let encoder6 = MaxPool2D<Float>(poolSize: (2, 2), stride=2, padding="same")

    let decoder1 = Conv2d(filterShape=(3, 3, 8, 8), padding="same", activation= relu)
    let decoder2 = UpSampling2D<Float>(size: 2)
    let decoder3 = Conv2d(filterShape=(3, 3, 8, 8), padding="same", activation= relu)
    let decoder4 = UpSampling2D<Float>(size: 2)
    let decoder5 = Conv2d(filterShape=(3, 3, 8, 16), activation= relu)
    let decoder6 = UpSampling2D<Float>(size: 2)

    let output = Conv2d(filterShape=(3, 3, 16, 1), padding="same", activation= sigmoid)

    
    override _.forward(input) =
        let resize = input.reshape([batchSize, 28, 28, 1])
        let encoder = resize |> encoder1,
            encoder2, encoder3, encoder4, encoder5, encoder6)
        let decoder = encoder |> decoder1,
            decoder2, decoder3, decoder4, decoder5, decoder6)
        return output(decoder).reshape([batchSize, imageHeight * imageWidth])



let model = Autoencoder2D()
let optimizer = AdaDelta(model)

// Training loop
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() = 
    vae.mode <- Mode.Train
    for batch in epochBatches do
        let x = batch.data

        let δmodel = TensorFlow.gradient(at: model) =  model -> Tensor<Float> in
            let image = model(x)
            return meanSquaredError(predicted: image, expected: x)


        optimizer.update(&model, along: δmodel)


    vae.mode <- Mode.Eval
    let testLossSum: double = 0
    let testBatchCount = 0
    for batch in dataset.validation do
        let sampleImages = batch.data
        let testImages = model(sampleImages)

        try
            try saveImage(
                sampleImages[0..<1], shape=[imageWidth; imageHeight], format: .grayscale,
                directory: outputFolder, name= "epoch-\(epoch)-input")
            try saveImage(
                testImages[0..<1], shape=[imageWidth; imageHeight], format: .grayscale,
                directory: outputFolder, name= "epoch-\(epoch)-output")
        with e ->
            print("Could not save image with error: \(error)")


        testLossSum <- testLossSum + meanSquaredError(predicted: testImages, expected: sampleImages).scalarized()
        testBatchCount <- testBatchCount + 1


    print(
        """
        [Epoch \(epoch)] \
        Loss: \(testLossSum / double(testBatchCount))
        """
    )

