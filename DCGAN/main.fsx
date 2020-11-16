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
#r @"..\bin\Debug\netcoreapp3.0\publish\DiffSharp.Core.dll"
#r @"..\bin\Debug\netcoreapp3.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\bin\Debug\netcoreapp3.0\publish\Library.dll"
#r @"System.Runtime.Extensions.dll"

open Datasets
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Util

let batchSize = 512
let epochCount = 20
let zDim = 100
let outputFolder = "./output/"

let dataset = MNIST(batchSize= batchSize,  flattening=false, normalizing=true)

type Generator() = 
    inherit Model()
    let flatten = Flatten()

    let dense1 = Linear(inFeatures=zDim, outFeatures=7 * 7 * 256)
    let batchNorm1 = BatchNorm2d(numFeatures=7 * 7 * 256)
    let transConv2D1 = ConvTranspose2d(256, 128, 5, stride=1, padding=5/2)
    let batchNorm2 = BatchNorm2d(numFeatures=7 * 7 * 128)
    let transConv2D2 = ConvTranspose2d(128, 64, kernelSize=5,stride=2,padding=5/2)
    let batchNorm3 = BatchNorm2d(numFeatures=14 * 14 * 64)
    let transConv2D3 = ConvTranspose2d(64, 1 , 5, stride=2, padding=5/2)

    override _.forward(input) =
        let x1 = input |> dense1.forward |> batchNorm1.forward |> dsharp.leakyRelu
        let x1Reshape = x1.view([-1; 7; 7; 256])
        let x2 = x1Reshape |> transConv2D1.forward |> flatten.forward |> batchNorm2.forward |> dsharp.leakyRelu
        let x2Reshape = x2.view([-1; 7; 7; 128])
        let x3 = x2Reshape |> transConv2D2.forward |> flatten.forward |> batchNorm3.forward |> dsharp.leakyRelu
        let x3Reshape = x3.view([-1; 14; 14; 64])
        x3Reshape |> transConv2D3.forward |> tanh

let generatorLoss(fakeLabels: Tensor) =
    dsharp.sigmoidCrossEntropy(logits=fakeLabels,labels=dsharp.ones(fakeLabels.shape))

// MARK: Discriminator

type Discriminator() =
    inherit Model()
    let conv2D1 = Conv2d(1, 64, kernelSize=5, stride=2,padding=5/2)
    let dropout = Dropout2d(p=0.3)
    let conv2D2 = Conv2d(64, 128, kernelSize=5,stride=2,padding=5/2)
    let flatten = Flatten()
    let dense = Linear(inFeatures=6272, outFeatures=1)
    
    override _.forward(input) =
        let x1 = input |> conv2D1.forward |> dsharp.leakyRelu |> dropout.forward
        let x2 = x1 |> conv2D2.forward |> dsharp.leakyRelu |> dropout.forward
        x2 |> flatten.forward |> dense.forward

let discriminatorLoss(realLabels: Tensor, fakeLabels: Tensor) : Tensor =
    let realLoss = dsharp.sigmoidCrossEntropy(logits=realLabels, labels=dsharp.ones(realLabels.shape))
    let fakeLoss = dsharp.sigmoidCrossEntropy(logits=fakeLabels, labels=dsharp.zeros(fakeLabels.shape))
    realLoss + fakeLoss

// MARK: - Training

// Create instances of models.
let discriminator = Discriminator()
let generator = Generator()

// Define optimizers.
let optG = Adam(generator, learningRate=dsharp.scalar 0.0001)
let optD = Adam(discriminator, learningRate=dsharp.scalar 0.0001)

// Test noise so we can track progress.
let noise = dsharp.randn([1; zDim])

print("Begin training...")
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() = 
    model.mode <- Mode.Train
    for batch in epochBatches do
        let realImages = batch.data

        // Train generator.
        let noiseG = dsharp.randn([batchSize; zDim])
        let δgenerator = dsharp.grad(generator) =  generator -> Tensor<Float> in
            let fakeImages = generator(noiseG)
            let fakeLabels = discriminator(fakeImages)
            let loss = generatorLoss(fakeLabels: fakeLabels)
            return loss

        optG.update(&generator, along=δgenerator)

        // Train discriminator.
        let noiseD = dsharp.randn([batchSize; zDim])
        let fakeImages = generator(noiseD)

        let δdiscriminator = dsharp.grad(discriminator) =  discriminator -> Tensor<Float> in
            let realLabels = discriminator(realImages)
            let fakeLabels = discriminator(fakeImages)
            let loss = discriminatorLoss(realLabels: realLabels, fakeLabels: fakeLabels)
            return loss

        optD.update(&discriminator, along=δdiscriminator)

    // Test the networks.
    vae.mode <- Mode.Eval

    // Render images.
    let generatedImage = generator(noise)
    try saveImage(
        generatedImage, shape=[28; 28], format="grayscale", directory=outputFolder,
        name= "\(epoch)")

    // Print loss.
    let generatorLoss_ = generatorLoss(fakeLabels: generatedImage)
    print("epoch: \(epoch) | Generator loss: \(generatorLoss_)")

// Generate another image.
let noise1 = dsharp.randn([1; 100])
let generatedImage = generator(noise1)
dsharp.saveImage(generatedImage, shape=[28; 28], format="grayscale", directory=outputFolder, name= "final")
