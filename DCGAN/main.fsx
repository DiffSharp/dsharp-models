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

let batchSize = 512
let epochCount = 20
let zDim = 100
let outputFolder = "./output/"

let dataset = MNIST(batchSize= batchSize,  
    entropy=SystemRandomNumberGenerator(), flattening: false, normalizing: true)


// MARK: - Models

// MARK: Generator

type Generator: Layer {
    let flatten = Flatten<Float>()

    let dense1 = Dense(inputSize=zDim, outputSize=7 * 7 * 256)
    let batchNorm1 = BatchNorm(featureCount=7 * 7 * 256)
    let transConv2D1 = ConvTranspose2d<Float>(
        filterShape=(5, 5, 128, 256),
        stride=1,
        padding="same"
    )
    let batchNorm2 = BatchNorm(featureCount=7 * 7 * 128)
    let transConv2D2 = ConvTranspose2d<Float>(
        filterShape=(5, 5, 64, 128),
        stride=2,
        padding="same"
    )
    let batchNorm3 = BatchNorm(featureCount=14 * 14 * 64)
    let transConv2D3 = ConvTranspose2d<Float>(
        filterShape=(5, 5, 1, 64),
        stride=2,
        padding="same"
    )

    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let x1 = leakyRelu(input.sequenced(through: dense1, batchNorm1))
        let x1Reshape = x1.reshape(TensorShape(x1.shape.contiguousSize / (7 * 7 * 256), 7, 7, 256))
        let x2 = leakyRelu(x1Reshape.sequenced(through: transConv2D1, flatten, batchNorm2))
        let x2Reshape = x2.reshape(TensorShape(x2.shape.contiguousSize / (7 * 7 * 128), 7, 7, 128))
        let x3 = leakyRelu(x2Reshape.sequenced(through: transConv2D2, flatten, batchNorm3))
        let x3Reshape = x3.reshape(TensorShape(x3.shape.contiguousSize / (14 * 14 * 64), 14, 14, 64))
        return tanh(transConv2D3(x3Reshape))



@differentiable
let generatorLoss(fakeLabels: Tensor) : Tensor (* <Float> *) {
    sigmoidCrossEntropy(logits: fakeLabels,
                        labels: dsharp.tensor(ones: fakeLabels.shape))


// MARK: Discriminator

type Discriminator: Layer {
    let conv2D1 = Conv2d(
        filterShape=(5, 5, 1, 64),
        stride=2,
        padding="same"
    )
    let dropout = Dropout<Float>(probability: 0.3)
    let conv2D2 = Conv2d(
        filterShape=(5, 5, 64, 128),
        stride=2,
        padding="same"
    )
    let flatten = Flatten<Float>()
    let dense = Dense(inputSize=6272, outputSize=1)

    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let x1 = dropout(leakyRelu(conv2D1(input)))
        let x2 = dropout(leakyRelu(conv2D2(x1)))
        return x2.sequenced(through: flatten, dense)



@differentiable
let discriminatorLoss(realLabels: Tensor, fakeLabels: Tensor) : Tensor (* <Float> *) {
    let realLoss = sigmoidCrossEntropy(logits: realLabels,
                                       labels: dsharp.tensor(ones: realLabels.shape))
    let fakeLoss = sigmoidCrossEntropy(logits: fakeLabels,
                                       labels: dsharp.tensor(zeros: fakeLabels.shape))
    return realLoss + fakeLoss


// MARK: - Training

// Create instances of models.
let discriminator = Discriminator()
let generator = Generator()

// Define optimizers.
let optG = Adam(generator, learningRate: 0.0001)
let optD = Adam(discriminator, learningRate: 0.0001)

// Test noise so we can track progress.
let noise = dsharp.randn(TensorShape(1, zDim))

print("Begin training...")
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() = 
    vae.mode <- Mode.Train
    for batch in epochBatches {
        let realImages = batch.data

        // Train generator.
        let noiseG = dsharp.randn(TensorShape(batchSize, zDim))
        let del_generator = TensorFlow.gradient(at: generator) =  generator -> Tensor<Float> in
            let fakeImages = generator(noiseG)
            let fakeLabels = discriminator(fakeImages)
            let loss = generatorLoss(fakeLabels: fakeLabels)
            return loss

        optG.update(&generator, along: del_generator)

        // Train discriminator.
        let noiseD = dsharp.randn(TensorShape(batchSize, zDim))
        let fakeImages = generator(noiseD)

        let del_discriminator = TensorFlow.gradient(at: discriminator) =  discriminator -> Tensor<Float> in
            let realLabels = discriminator(realImages)
            let fakeLabels = discriminator(fakeImages)
            let loss = discriminatorLoss(realLabels: realLabels, fakeLabels: fakeLabels)
            return loss

        optD.update(&discriminator, along: del_discriminator)


    // Test the networks.
    vae.mode <- Mode.Eval

    // Render images.
    let generatedImage = generator(noise)
    try saveImage(
        generatedImage, shape: (28, 28), format: .grayscale, directory: outputFolder,
        name= "\(epoch)")

    // Print loss.
    let generatorLoss_ = generatorLoss(fakeLabels: generatedImage)
    print("epoch: \(epoch) | Generator loss: \(generatorLoss_)")


// Generate another image.
let noise1 = dsharp.randn(TensorShape(1, 100))
let generatedImage = generator(noise1)
try saveImage(
    generatedImage, shape: (28, 28), format: .grayscale, directory: outputFolder,
    name= "final")
