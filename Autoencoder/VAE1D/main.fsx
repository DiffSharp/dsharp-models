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

// Based on the paper: "Auto-Encoding Variational Bayes"
// by Diederik P Kingma and Max Welling
// Reference implementation: https://github.com/pytorch/examples/blob/master/vae/main.py

open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data
open Datasets

let epochCount = 10
let batchSize = 128
let imageHeight = 28
let imageWidth = 28

let outputFolder = "./output/"
let dataset = MNIST(batchSize= batchSize, entropy=SystemRandomNumberGenerator(), flattening=true)

let inputDim = 784  // 28*28 for any MNIST
let hiddenDim = 400
let latentDim = 20

// Variational Autoencoder
type VAE() =
    inherit Model()
    // Encoder
    let encoderDense1 = Dense(inputSize= inputDim, outputSize=hiddenDim, activation= dsharp.relu)
    let encoderDense2_1 = Dense(inputSize=hiddenDim, outputSize=latentDim)
    let encoderDense2_2 = Dense(inputSize=hiddenDim, outputSize=latentDim)

    let decoderDense1 = Dense(inputSize= latentDim, outputSize=hiddenDim, activation= dsharp.relu)
    let decoderDense2 = Dense(inputSize=hiddenDim, outputSize=inputDim)

    member _.call(input: Tensor) =
        // Encode
        let intermediateInput = encoderDense1.forward(input)
        let mu = encoderDense2_1.forward(intermediateInput)
        let logVar = encoderDense2_2.forward(intermediateInput)

        // Re-parameterization trick
        let std = exp(0.5 * logVar)
        let epsilon = dsharp.randn(std.shape)
        let z = mu + epsilon * std

        // Decode
        let output = z --> decoderDense1 --> decoderDense2
        output, mu, logVar

    override t.forward(input: Tensor) =
        let output, mu, logVar = t.call(input)
        output


let vae = VAE()
let optimizer = Adam(vae, lr = dsharp.scalar 1e-3)

// Loss function: sum of the KL divergence of the embeddings and the cross entropy loss between the input and it's reconstruction. 
let vaeLossFunction(input: Tensor, output: Tensor, mu: Tensor, logVar: Tensor) : Tensor (* <Float> *) =
    let crossEntropy : Tensor =  failwith "tbd" // sigmoidCrossEntropy(logits: output, labels: input, reduction: _sum)
    let klDivergence = -0.5 * (1 + logVar - mu ** 2 - exp(logVar)).sum()
    crossEntropy + klDivergence


// Training loop
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() do
    vae.mode <- Mode.Train
    for batch in epochBatches do
        let x = batch.data
        vae.reverseDiff()
        let output, mu, logVar = vae(x)
        vaeLossFunction(x, output, mu, logVar)

        optimizer.update(&vae, along: del_model)


    vae.mode <- Mode.Eval
    let mutable testLossSum: double = 0.0
    let mutable testBatchCount = 0
    for batch in dataset.validation do
        let sampleImages = batch.data
        let testImages, testMus, testLogVars = vae.call(sampleImages)
        //if epoch = 0 || (epoch + 1) % 10 = 0 then
        //    try
        //        try saveImage(
        //            sampleImages[0..<1], shape: (imageWidth, imageHeight), format: .grayscale,
        //            directory: outputFolder, name= "epoch-\(epoch)-input")
        //        try saveImage(
        //            testImages[0..<1], shape: (imageWidth, imageHeight), format: .grayscale,
        //            directory: outputFolder, name= "epoch-\(epoch)-output")
        //    with e ->
        //        print("Could not save image with error: \(error)")

        //testLossSum <- testLossSum + vaeLossFunction(
        //    input: sampleImages, output: testImages, mu: testMus, logVar: testLogVars).scalarized() / double(batchSize)
        testBatchCount <- testBatchCount + 1

    print(
        """
        [Epoch \(epoch)] \
        Loss: \(testLossSum / double(testBatchCount))
        """
    )

