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
open Datasets


open DiffSharp

let epochCount = 10
let batchSize = 32
let outputFolder = "./output/"
let imageHeight = 28
let imageWidth = 28
let imageSize = imageHeight * imageWidth
let latentSize = 64

// Models

type Generator: Layer {
    let dense1 = Dense(
        inputSize= latentSize, outputSize=latentSize * 2,
        activation= { leakyRelu($0))

    let dense2 = Dense(
        inputSize= latentSize * 2, outputSize=latentSize * 4,
        activation= { leakyRelu($0))

    let dense3 = Dense(
        inputSize= latentSize * 4, outputSize=latentSize * 8,
        activation= { leakyRelu($0))

    let dense4 = Dense(
        inputSize= latentSize * 8, outputSize=imageSize,
        activation= tanh)

    let batchnorm1 = BatchNorm(featureCount=latentSize * 2)
    let batchnorm2 = BatchNorm(featureCount=latentSize * 4)
    let batchnorm3 = BatchNorm(featureCount=latentSize * 8)

    
    override _.forward(input) =
        let x1 = batchnorm1(dense1(input))
        let x2 = batchnorm2(dense2(x1))
        let x3 = batchnorm3(dense3(x2))
        return dense4(x3)



type Discriminator: Layer {
    let dense1 = Dense(
        inputSize= imageSize, outputSize=256,
        activation= { leakyRelu($0))

    let dense2 = Dense(
        inputSize= 256, outputSize=64,
        activation= { leakyRelu($0))

    let dense3 = Dense(
        inputSize= 64, outputSize=16,
        activation= { leakyRelu($0))

    let dense4 = Dense(
        inputSize= 16, outputSize=1,
        activation= identity)

    
    override _.forward(input) =
        input |> dense1, dense2, dense3, dense4)



// Loss functions


let generatorLoss(fakeLogits: Tensor) : Tensor =
    dsharp.sigmoidCrossEntropy(
        logits: fakeLogits,
        labels=dsharp.ones(fakeLogits.shape))



let discriminatorLoss(realLogits: Tensor, fakeLogits: Tensor) : Tensor =
    let realLoss = dsharp.sigmoidCrossEntropy(
        logits: realLogits,
        labels=dsharp.ones(realLogits.shape))
    let fakeLoss = dsharp.sigmoidCrossEntropy(
        logits: fakeLogits,
        labels=dsharp.zeros(fakeLogits.shape))
    return realLoss + fakeLoss


/// Returns `size` samples of noise vector.
let sampleVector(size: int) : Tensor =
    dsharp.tensor(randomNormal: [size, latentSize])


let dataset = MNIST(batchSize= batchSize,  
    flattening=true, normalizing=true)

let generator = Generator()
let discriminator = Discriminator()

let optG = Adam(generator, learningRate: 2e-4, beta1=dsharp.scalar(0.5))
let optD = Adam(discriminator, learningRate: 2e-4, beta1=dsharp.scalar(0.5))

// Noise vectors and plot function for testing
let testImageGridSize = 4
let testVector = sampleVector(size: testImageGridSize * testImageGridSize)

let saveImageGrid(_ testImage: Tensor, name: string) =
    let gridImage = testImage.reshape(
        [
            testImageGridSize, testImageGridSize,
            imageHeight, imageWidth,
        ])
    // Add padding.
    gridImage = gridImage.pad(forSizes: [(0, 0), (0, 0), (1, 1), (1, 1)], with: 1)
    // Transpose to create single image.
    gridImage = gridImage.transposed(permutation: [0, 2, 1, 3])
    gridImage = gridImage.reshape(
        [
            (imageHeight + 2) * testImageGridSize,
            (imageWidth + 2) * testImageGridSize,
        ])
    // Convert [-1, 1] range to [0, 1] range.
    gridImage = (gridImage + 1) / 2

    try saveImage(
        gridImage, shape=gridImage.shape.[0..1], format: .grayscale,
        directory: outputFolder, name= name)


print("Start training...")

// Start training loop.
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() = 
    // Start training phase.
    vae.mode <- Mode.Train
    for batch in epochBatches do
        // Perform alternative update.
        // Update generator.
        let vec1 = sampleVector(size: batchSize)

        let del_generator = TensorFlow.gradient(at: generator) =  generator -> Tensor<Float> in
            let fakeImages = generator(vec1)
            let fakeLogits = discriminator(fakeImages)
            let loss = generatorLoss(fakeLogits: fakeLogits)
            return loss

        optG.update(&generator, along: del_generator)

        // Update discriminator.
        let realImages = batch.data
        let vec2 = sampleVector(size: batchSize)
        let fakeImages = generator(vec2)

        let del_discriminator = TensorFlow.gradient(at: discriminator) =  discriminator -> Tensor<Float> in
            let realLogits = discriminator(realImages)
            let fakeLogits = discriminator(fakeImages)
            let loss = discriminatorLoss(realLogits: realLogits, fakeLogits: fakeLogits)
            return loss

        optD.update(&discriminator, along: del_discriminator)


    // Start inference phase.
    vae.mode <- Mode.Eval
    let testImage = generator(testVector)

    try
        try saveImageGrid(testImage, name= "epoch-\(epoch)-output")
    with
        print("Could not save image grid with error: \(error)")


    let lossG = generatorLoss(fakeLogits: testImage)
    print("[Epoch: \(epoch)] Loss-G: \(lossG)")
