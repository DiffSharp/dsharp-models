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
open DiffSharp.ShapeChecking

[<ShapeCheck(16,32,3,1)>]
type Conv(inChannels: int, outChannels: int, kernelSize: int, ?stride: int) =
    inherit Model()
    let stride = defaultArg stride 1

    let batchNorm = BatchNorm2d(numFeatures=inChannels)
    let conv = Conv2d(inChannels, outChannels, kernelSize, strides = [stride; stride], padding= kernelSize/2 (* "same"  *))

    [<ShapeCheck("N,inChannels,H,W", ReturnShape="N,outChannels,H,W")>]
    override _.forward(input) =
        conv.forward(dsharp.relu(batchNorm.forward(input)))

/// A pair of a 1x1 `Conv` layer and a 3x3 `Conv` layer.
[<ShapeCheck("16",32)>]
type ConvPair(inChannels: int, growthRate: int) =
    inherit Model()
    let conv1x1 = Conv(kernelSize=1, inChannels=inChannels, outChannels=inChannels * 2 )
    let conv3x3 = Conv(kernelSize=3, inChannels=inChannels * 2, outChannels=growthRate)
        
    [<ShapeCheck("N,inChannels,H,W", ReturnShape="N,inChannels+growthRate,H,W")>]
    override _.forward(input) =
        let conv1Output = conv1x1.forward(input)
        let conv3Output = conv3x3.forward(conv1Output)
        dsharp.cat ([| conv3Output; input |], 1)

// Original Paper:
// Densely Connected Convolutional Networks
// Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
// https://arxiv.org/pdf/1608.06993.pdf

[<ShapeCheck(5,16,32)>]
type DenseBlock(repetitionCount: int, inChannels:int, ?growthRate: int) =
    inherit Model()
    let growthRate = defaultArg growthRate 32
    let pairs = 
        [| for i in 0..repetitionCount-1 do
            let filterCount = inChannels + i * growthRate
            ConvPair(inChannels=filterCount, growthRate=growthRate) |]
    
    [<ShapeCheck("N,inChannels,H,W", ReturnShape="N,inChannels+repetitionCount*growthRate,H,W")>]
    override _.forward(input) =
        (input, pairs) ||> Array.fold (fun last layer -> layer.forward last) 

[<ShapeCheck(16)>]
type TransitionLayer(inChannels:int) = 
    inherit Model()
    let conv = Conv(kernelSize=1, inChannels=inChannels, outChannels=inChannels / 2)
    // padding = ((stride - 1) + dilation * (kernel_size - 1)) / 2
    let pool = AvgPool2d(kernelSize=2, stride=2, padding=1 (* "same " *))
    
    // TODO, check this with original python source - "same" is not statically computable for non-unit stride
    [<ShapeCheck("N,inChannels,H,W", ReturnShape="N,inChannels/2,1+H/2,1+W/2")>] 
    override _.forward(input) =
        let res = input |> conv.forward |> pool.forward
        res

[<ShapeCheck(16)>]
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

    [<ShapeCheck("N,3,H,W", ReturnShape="N,classCount")>]
    override _.forward(input) =
        let inputLayer = input |> conv.forward |> maxpool.forward
        let level1 = inputLayer |> denseBlock1.forward |> transitionLayer1.forward
        let level2 = level1 |> denseBlock2.forward |> transitionLayer2.forward
        let level3 = level2 |> denseBlock3.forward |> transitionLayer3.forward
        let output = level3 |> denseBlock4.forward |> globalAvgPool.forward |> dense.forward
        output
        