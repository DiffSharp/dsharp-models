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

#load @"Data\Dataset.fsx"
#load @"Models\Layers.fsx"
#load @"Models\Generator.fsx"
#load @"Models\Discriminator.fsx"

open System.IO
open DiffSharp
open DiffSharp.Optim
open Datasets
open Dataset
open Layers
open Generator
open Discriminator

//let options = Options.parseOrExit()

let dataset = CycleGANDataset(trainBatchSize= 1, testBatchSize= 1)

let generatorG = ResNetGenerator(inputChannels=3, outputChannels=3, blocks=9, ngf=64)
let generatorF = ResNetGenerator(inputChannels=3, outputChannels=3, blocks=9, ngf=64)
let discriminatorX = NetD(inChannels=3, lastConvFilters=64)
let discriminatorY = NetD(inChannels=3, lastConvFilters=64)

let optimizerGF = Adam(generatorF, learningRate=dsharp.scalar(0.0002), beta1=dsharp.scalar(0.5))
let optimizerGG = Adam(generatorG, learningRate=dsharp.scalar(0.0002), beta1=dsharp.scalar(0.5))
let optimizerDX = Adam(discriminatorX, learningRate=dsharp.scalar(0.0002), beta1=dsharp.scalar(0.5))
let optimizerDY = Adam(discriminatorY, learningRate=dsharp.scalar(0.0002), beta1=dsharp.scalar(0.5))

let epochCount = options.epochs
let lambdaL1 = Tensorf(10)
let _zeros = Tensorf.zero
let _ones = Tensorf.one

let step = 0

let validationImage = (fst dataset.TrainingSamples.[0]).unsqueeze(0)
let validationImageURL = __SOURCE_DIRECTORY__ </> ("sample.jpg")

// MARK: Train

for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() = 
    print("Epoch \(epoch) started at: \(Date())")
    vae.mode <- Mode.Train
    
    for batch in epochBatches do
        vae.mode <- Mode.Train
        
        let inputX = batch.domainA
        let inputY = batch.domainB

        // we do it outside of GPU scope so that dataset shuffling happens on CPU side
        let concatanatedImages = inputX.concatenated(inputY)

        let scaledImages = resize(images: concatanatedImages, size: (286, 286), method: .nearest)
        let croppedImages = scaledImages.slice(lowerBounds: Tensor (*<int32>*)([0, int32(random() % 30), int32(random() % 30), 0]),
                                               sizes: [2, 256, 256, 3])
        if Bool.random() = 
            croppedImages = croppedImages.reversed(inAxes: 2)

        let realX = croppedImages[0].unsqueeze(0)
        let realY = croppedImages[1].unsqueeze(0)

        let onesd = _ones.expand([1, 30, 30, 1])
        let zerosd = _zeros.expand([1, 30, 30, 1])

        let _fakeX = Tensorf.zero
        let _fakeY = Tensorf.zero

        let (gLoss, del_generatorG) = valueWithGradient(at: generatorG) =  g -> Tensorf in
            let fakeY = g(realX)
            let cycledX = generatorF(fakeY)
            let fakeX = generatorF(realY)
            let cycledY = g(fakeX)

            let cycleConsistencyLoss = (abs(realX - cycledX).mean() +
                abs(realY - cycledY).mean()) * lambdaL1

            let discFakeY = discriminatorY(fakeY)
            let generatorLoss = dsharp.sigmoidCrossEntropy(logits=discFakeY, labels: onesd)

            let sameY = g(realY)
            let identityLoss = abs(sameY - realY).mean() * lambdaL1 * 0.5

            let totalLoss = cycleConsistencyLoss + generatorLoss + identityLoss

            _fakeX = fakeX

            totalLoss


        let (fLoss, del_generatorF) = valueWithGradient(at: generatorF) =  g -> Tensorf in
            let fakeX = g(realY)
            let cycledY = generatorG(fakeX)
            let fakeY = generatorG(realX)
            let cycledX = g(fakeY)

            let cycleConsistencyLoss = (abs(realY - cycledY).mean()
                + abs(realX - cycledX).mean()) * lambdaL1

            let discFakeX = discriminatorX(fakeX)
            let generatorLoss = dsharp.sigmoidCrossEntropy(logits=discFakeX, labels: onesd)

            let sameX = g(realX)
            let identityLoss = abs(sameX - realX).mean() * lambdaL1 * 0.5

            let totalLoss = cycleConsistencyLoss + generatorLoss + identityLoss

            _fakeY = fakeY
            totalLoss


        let (xLoss, del_discriminatorX) = valueWithGradient(at: discriminatorX) =  d -> Tensorf in
            let discFakeX = d(_fakeX)
            let discRealX = d(realX)

            let totalLoss = 0.5 * (sigmoidCrossEntropy(logits=discFakeX, labels: zerosd)
                + dsharp.sigmoidCrossEntropy(logits=discRealX, labels: onesd))

            totalLoss


        let (yLoss, del_discriminatorY) = valueWithGradient(at: discriminatorY) =  d -> Tensorf in
            let discFakeY = d(_fakeY)
            let discRealY = d(realY)

            let totalLoss = 0.5 * (sigmoidCrossEntropy(logits=discFakeY, labels: zerosd)
                + dsharp.sigmoidCrossEntropy(logits=discRealY, labels: onesd))

            totalLoss


        optimizerGG.update(&generatorG, along: del_generatorG)
        optimizerGF.update(&generatorF, along: del_generatorF)
        optimizerDX.update(&discriminatorX, along: del_discriminatorX)
        optimizerDY.update(&discriminatorY, along: del_discriminatorY)

        // MARK: Inference

        if step % options.sampleLogPeriod = 0 then
            vae.mode <- Mode.Eval
            
            let fakeSample = generatorG(validationImage) * 0.5 + 0.5

            let fakeSampleImage = Image(tensor: fakeSample[0] * 255)
            fakeSampleImage.save(validationImageURL, format="rgb")

            print("GeneratorG loss: \(gLoss.scalars[0])")
            print("GeneratorF loss: \(fLoss.scalars[0])")
            print("DiscriminatorX loss: \(xLoss.scalars[0])")
            print("DiscriminatorY loss: \(yLoss.scalars[0])")


        step <- step + 1

// MARK: Final test

let aResultsFolder = Directory.Create(__SOURCE_DIRECTORY__ + "/testA_results")
let bResultsFolder = Directory.Create(__SOURCE_DIRECTORY__ + "/testB_results")

let testStep = 0
for testBatch in dataset.testing do
    let realX = testBatch.domainA
    let realY = testBatch.domainB

    let fakeY = generatorG(realX)
    let fakeX = generatorF(realY)

    let resultX = realX.concatenated(fakeY, alongAxis: 2) * 0.5 + 0.5
    let resultY = realY.concatenated(fakeX, alongAxis: 2) * 0.5 + 0.5

    let imageX = Image(tensor: resultX[0] * 255)
    let imageY = Image(tensor: resultY[0] * 255)

    imageX.save(aResultsFolder </> ("\(String(testStep)).jpg", isDirectory: false),
                format="rgb")
    imageY.save(bResultsFolder </> ("\(String(testStep)).jpg", isDirectory: false),
                format="rgb")

    testStep <- testStep + 1

