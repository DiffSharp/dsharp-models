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

module Models.ImageClassification.VGG

open DiffSharp
open DiffSharp.Model

// Original Paper:
// "Very Deep Convolutional Networks for Large-Scale Image Recognition"
// Karen Simonyan, Andrew Zisserman
// https://arxiv.org/abs/1409.1556

type VGGBlock(featureCounts, blockCount: int) =
    inherit Model()
    let maxpool = MaxPool2d(kernelSize=2, stride=2)
    let (featureCounts0, featureCounts1, featureCounts2, featureCounts3) = featureCounts

    let blocks = 
        [| Conv2d(featureCounts0, featureCounts1, kernelSize=3, padding=1 (* "same " *), activation=dsharp.relu) 
           for _ in 1..blockCount-1 do
              Conv2d(featureCounts2, featureCounts3, kernelSize=3,
                padding=1 (* "same " *),
                activation=dsharp.relu) |]
    
    override _.forward(input) =
        (input, blocks) ||> Array.fold (fun last layer -> layer.forward last) 
        |> maxpool.forward

type VGG16(?classCount: int) =
    inherit Model()
    let classCount = defaultArg classCount 1000

    let layer1 = VGGBlock(featureCounts=(3, 64, 64, 64), blockCount=2)
    let layer2 = VGGBlock(featureCounts=(64, 128, 128, 128), blockCount=2)
    let layer3 = VGGBlock(featureCounts=(128, 256, 256, 256), blockCount=3)
    let layer4 = VGGBlock(featureCounts=(256, 512, 512, 512), blockCount=3)
    let layer5 = VGGBlock(featureCounts=(512, 512, 512, 512), blockCount=3)
    let output = Linear(inFeatures=4096, outFeatures=classCount)
    let flatten = Flatten()
    let dense1 = Linear(inFeatures=512 * 7 * 7, outFeatures=4096, activation=dsharp.relu)
    let dense2 = Linear(inFeatures=4096, outFeatures=4096, activation=dsharp.relu)

    override _.forward(input) =
        let backbone = input |> layer1.forward |> layer2.forward |> layer3.forward |> layer4.forward |> layer5.forward
        backbone |> flatten.forward |> dense1.forward |> dense2.forward |> output.forward

type VGG19(?classCount: int) =
    inherit Model()
    let classCount = defaultArg classCount 1000

    let layer1 = VGGBlock(featureCounts=(3, 64, 64, 64), blockCount=2)
    let layer2 = VGGBlock(featureCounts=(64, 128, 128, 128), blockCount=2)
    let layer3 = VGGBlock(featureCounts=(128, 256, 256, 256), blockCount=4)
    let layer4 = VGGBlock(featureCounts=(256, 512, 512, 512), blockCount=4)
    let layer5 = VGGBlock(featureCounts=(512, 512, 512, 512), blockCount=4)
    let flatten = Flatten()
    let dense1 = Linear(inFeatures=512 * 7 * 7, outFeatures=4096, activation=dsharp.relu)
    let dense2 = Linear(inFeatures=4096, outFeatures=4096, activation=dsharp.relu)
    let output = Linear(inFeatures=4096, outFeatures=classCount)
    
    override _.forward(input) =
        let backbone = input |> layer1.forward |> layer2.forward |> layer3.forward |> layer4.forward |> layer5.forward
        backbone |> flatten.forward |> dense1.forward |> dense2.forward |> output.forward
