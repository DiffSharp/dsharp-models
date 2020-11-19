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
#r @"System.Runtime.Extensions.dll"

open Datasets

open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open TrainingLoop

let epochCount = 10
let batchSize = 100
let imageHeight = 28
let imageWidth = 28

let dataset = FashionMNIST(batchSize=batchSize, flattening=true)

// An autoencoder.
let autoencoder = 
    Sequential (
          // The encoder.
          Linear(inFeatures=imageHeight * imageWidth, outFeatures=128, activation=dsharp.relu),
          Linear(inFeatures=128, outFeatures=64, activation=dsharp.relu),
          Linear(inFeatures=64, outFeatures=12, activation=dsharp.relu),
          Linear(inFeatures=12, outFeatures=3, activation=dsharp.relu),
          // The decoder.
          Linear(inFeatures=3, outFeatures=12, activation=dsharp.relu),
          Linear(inFeatures=12, outFeatures=64, activation=dsharp.relu),
          Linear(inFeatures=64, outFeatures=128, activation=dsharp.relu),
          Linear(inFeatures=128, outFeatures=imageHeight * imageWidth, activation=dsharp.tanh)
    )

let optimizer = RMSProp(autoencoder)

/// Saves a validation input and an output image once per epoch;
/// it's ensured that each epoch will save different images as long as 
/// count of epochs is less or equal than count of images.
/// 
/// It's defined as a callback registered into TrainingLoop.
let saveImage(loop: TrainingLoop, event: TrainingLoopEvent) =
  if event <> "inferencePredictionEnd" then () else

  let batchIndex = loop.batchIndex
  let batchCount = loop.batchCount
  let epochIndex = loop.epochIndex
  let epochCount = loop.epochCount
  let input = loop.lastStepInput
  let output = loop.lastStepOutput
  let imageCount = batchCount * batchSize
  let selectedImageGlobalIndex = epochIndex * (imageCount / epochCount)
  let selectedBatchIndex = selectedImageGlobalIndex / batchSize

  if batchIndex <> selectedBatchIndex then () else

  let outputFolder = "./output/"
  let selectedImageBatchLocalIndex = selectedImageGlobalIndex % batchSize
  dsharp.saveImage(
    input.[selectedImageBatchLocalIndex..selectedImageBatchLocalIndex].view([-1;imageWidth; imageHeight]),
    //format="grayscale",
    fileName= (outputFolder </> $"epoch-{epochIndex + 1}-of-{epochCount}-input"))
  dsharp.saveImage(
    output.[selectedImageBatchLocalIndex..selectedImageBatchLocalIndex].view([-1;imageWidth; imageHeight]),
    //format="grayscale",
    fileName= (outputFolder </> $"epoch-{epochIndex + 1}-of-{epochCount}-output"))


let trainingLoop =
    TrainingLoop(
       training=dataset.training |> Seq.map (Array.map (fun t -> LabeledData(t.data, label: t.data))),
       validation=dataset.validation |> Seq.map (Array.map (fun t -> LabeledData(t.data, t.data))),
       optimizer=optimizer,
       lossFunction=meanSquaredError,
       callbacks=[saveImage])

trainingLoop.fit(autoencoder, epoch=epochCount)
