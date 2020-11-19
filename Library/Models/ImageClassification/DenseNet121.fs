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

module Models.ImageClassification.DenseNet121

open DiffSharp
open DiffSharp.Model

type Conv(inChannels: int, outChannels: int, kernelSize: int, ?stride: int) =
    inherit Model()
    let stride = defaultArg stride 1

    let batchNorm = BatchNorm2d(numFeatures=inChannels)
    let conv = Conv2d(inChannels, outChannels, kernelSize, strides = [stride; stride], padding= kernelSize/2 (* "same"  *))

    override _.forward(input) =
        conv.forward(dsharp.relu(batchNorm.forward(input)))

/// A pair of a 1x1 `Conv` layer and a 3x3 `Conv` layer.
type ConvPair(inChannels: int, growthRate: int) =
    inherit Model()
    let conv1x1 = Conv(kernelSize=1, inChannels=inChannels, outChannels=inChannels * 2 )
    let conv3x3 = Conv(kernelSize=3, inChannels=inChannels * 2, outChannels=growthRate)
        
    override _.forward(input) =
        let conv1Output = conv1x1.forward(input)
        let conv3Output = conv3x3.forward(conv1Output)
        dsharp.cat [| conv3Output; input |]

// Original Paper:
// Densely Connected Convolutional Networks
// Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
// https://arxiv.org/pdf/1608.06993.pdf

type DenseBlock(repetitionCount: int, inChannels:int, ?growthRate: int) =
    inherit Model()
    let growthRate = defaultArg growthRate 32
    let pairs = 
        [| for i in 0..repetitionCount-1 do
            let filterCount = inChannels + i * growthRate
            ConvPair(inChannels=filterCount, growthRate=growthRate) |]
    
    override _.forward(input) =
        (input, pairs) ||> Array.fold (fun last layer -> layer.forward last) 

type TransitionLayer(inChannels:int) = 
    inherit Model()
    let conv = Conv(kernelSize=1, inChannels=inChannels, outChannels=inChannels / 2)
    let pool = AvgPool2d(kernelSize=2, stride=2, padding=1 (* "same " *))
    
    override _.forward(input) =
        input |> conv.forward |> pool.forward

type DenseNet121(classCount: int) = 
    inherit Model()
    let conv = Conv2d(kernelSize=7, stride=2, inChannels=3, outChannels=64)
    let maxpool = MaxPool2d(kernelSize=3,stride=2,padding=3/2 (* "same " *))
    let denseBlock1 = DenseBlock(repetitionCount=6, inChannels=64)
    let transitionLayer1 = TransitionLayer(inChannels=256)
    let denseBlock2 = DenseBlock(repetitionCount=12, inChannels=128)
    let transitionLayer2 = TransitionLayer(inChannels=512)
    let denseBlock3 = DenseBlock(repetitionCount=24, inChannels=256)
    let transitionLayer3 = TransitionLayer(inChannels=1024)
    let denseBlock4 = DenseBlock(repetitionCount=16, inChannels=512)
    let globalAvgPool = GlobalAvgPool2d()
    let dense = Linear(inFeatures=1024, outFeatures=classCount)

    override _.forward(input) =
        let inputLayer = input |> conv.forward |> maxpool.forward
        let level1 = inputLayer |> denseBlock1.forward |> transitionLayer1.forward
        let level2 = level1 |> denseBlock2.forward |> transitionLayer2.forward
        let level3 = level2 |> denseBlock3.forward |> transitionLayer3.forward
        let output = level3 |> denseBlock4.forward |> globalAvgPool.forward |> dense.forward
        output
