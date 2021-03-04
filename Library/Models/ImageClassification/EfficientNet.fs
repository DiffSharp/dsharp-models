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

module Models.ImageClassification.EfficientNet

open DiffSharp
open DiffSharp.Model
open DiffSharp.ShapeChecking

// Original Paper:
// "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
// Mingxing Tan, Quoc V. Le
// https://arxiv.org/abs/1905.11946
// Notes: Default baseline (B0) network, see table 1

/// some utility functions to help generate network variants
/// original: https://github.com/tensorflow/tpu/blob/d6f2ef3edfeb4b1c2039b81014dc5271a7753832/models/official/efficientnet/efficientnet_model.py#L138
let resizeDepth(blockCount: int, depth: double) =
    /// Multiply + round up the number of blocks based on depth multiplier
    let newChannels = depth * double(blockCount) |> ceil
    int newChannels

let makeDivisible(filter: int, width: double) =
    let divisor = 8.0
    /// Return a filter multiplied by width, rounded down and evenly divisible by the divisor
    let filterMult = double(filter) * width
    let filterAdd = double(filterMult) + (divisor / 2.0)
    let div = filterAdd / divisor |> floor
    let div = div * double(divisor)
    let mutable newChannels = max 1 (int div)
    if newChannels < int(0.9 * double filter) then
        newChannels <- newChannels + int(divisor)

    int newChannels

let roundFilterPair(f1, f2, width: double) =
    makeDivisible(f1, width),makeDivisible(f2, width)

[<ShapeCheck(32, 16)>]
type InitialMBConvBlock(inChannels: int, outChannels: int, width: double) = 
    inherit Model()

    let hiddenDimension, filterMult2 = roundFilterPair(inChannels, outChannels, width)
    let dConv = DepthwiseConv2d(hiddenDimension, 1, kernelSize=3, stride=1,padding=1 (* "same " *))
    let seReduceConv = Conv2d(hiddenDimension, makeDivisible(8, width), kernelSize=1, stride=1, padding=0 (* "same " *))
    let seExpandConv = Conv2d(makeDivisible(8, width), hiddenDimension, kernelSize=1, stride=1, padding=0 (* "same " *))
    let conv2 = Conv2d(hiddenDimension, filterMult2, kernelSize=1, stride=1, padding=0 (* "same " *))
    let batchNormDConv = BatchNorm2d(numFeatures=hiddenDimension)
    let seAveragePool = GlobalAvgPool2d()
    let batchNormConv2 = BatchNorm2d(numFeatures=filterMult2)

    [<ShapeCheck("N,inChannels,H,W", ReturnShape="N,outChannels,H,W")>]
    override _.forward(input) =
        let depthwise = input |> dConv.forward |> batchNormDConv.forward |> dsharp.silu 
        let seAvgPoolReshaped = depthwise |> seAveragePool.forward |> fun t -> t.view(Shape [ input.shapex.[0]; Int hiddenDimension; 1I; 1I ])
        let squeezeExcite =   
            seAvgPoolReshaped |> seReduceConv.forward  |> dsharp.silu |> seExpandConv.forward
            |> fun t -> depthwise * dsharp.sigmoid(t)
        squeezeExcite |> conv2.forward |> batchNormConv2.forward

[<ShapeCheck(32, 16, 1.0, 6, 2, 3)>]
type MBConvBlock(inChannels, outChannels, width: double, ?depthMultiplier, ?stride, ?kernel) =
    inherit Model()
    let kernel = defaultArg kernel 3
    let stride = defaultArg stride 1
    let depthMultiplier = defaultArg depthMultiplier 6
    let addResLayer = (inChannels = outChannels) && stride = 1
    
    let filterMult1, filterMult2 = roundFilterPair(inChannels, outChannels, width)
    let hiddenDimension = filterMult1 * depthMultiplier
    let reducedDimension = max 1 (int (filterMult1 / 4))
    let conv1 = Conv2d(filterMult1, hiddenDimension, kernelSize=1, stride=1,padding=0 (* "same " *))
    let dConv = DepthwiseConv2d(hiddenDimension, 1, kernelSize=kernel, stride=stride, padding=(stride-1 + kernel-1)/2 (* , paddings=(if strides = (1, 1) then ".same" else ".valid" *))
    let seReduceConv = Conv2d(hiddenDimension, reducedDimension, kernelSize=1, stride=1,padding=0 (* "same " *))
    let seExpandConv = Conv2d(reducedDimension, hiddenDimension, kernelSize=1, stride=1, padding=0 (* "same " *))
    let conv2 = Conv2d(hiddenDimension, filterMult2, 1, stride=1, padding=0 (* "same " *))
    let batchNormConv1 = BatchNorm2d(numFeatures=hiddenDimension)
    let zeroPad = ZeroPadding2d(1,1)
    let batchNormDConv = BatchNorm2d(numFeatures=hiddenDimension)
    let seAveragePool = GlobalAvgPool2d()
    let batchNormConv2 = BatchNorm2d(numFeatures=filterMult2)
    
    [<ShapeCheck("N,inChannels,H,W", ReturnShape="N,outChannels,1+(1+H/stride),1+(1+W/stride)")>]
    override _.forward(input) =
     
        let piecewise = input |> conv1.forward |> batchNormConv1.forward |> dsharp.silu
        
        let depthwise =
            if stride = 1 then
                piecewise |> dConv.forward |> batchNormDConv.forward |> dsharp.silu
            else
                piecewise |> zeroPad.forward |> dConv.forward |> batchNormDConv.forward |> dsharp.silu
                
        let seAvgPoolReshaped = 
            seAveragePool.forward(depthwise).view(Shape [
                input.shapex.[0]; Int hiddenDimension; 1I; 1I
            ]) 
        let squeezeExcite = depthwise * dsharp.sigmoid(seExpandConv.forward(dsharp.silu(seReduceConv.forward(seAvgPoolReshaped))))
        let piecewiseLinear = batchNormConv2.forward(conv2.forward(squeezeExcite))

        if addResLayer then
            input + piecewiseLinear
        else
            piecewiseLinear

[<ShapeCheck(32, 16, 1.0, 2)>]
type MBConvBlockStack(inChannels: int, outChannels: int,
        width: double,
        blockCount: int,
        depth: double,
        ?initialStride: int,
        ?kernel: int) =
    inherit Model()
    let initialStride = defaultArg initialStride 2
    let kernel = defaultArg kernel 3
    let blockMult = resizeDepth(blockCount, depth)
    let blocks = 
        [| MBConvBlock(inChannels, outChannels, width, stride=initialStride, kernel=kernel)
           for _ in 1..blockMult-1 do
              MBConvBlock(outChannels, outChannels,width, kernel=kernel) |]

    [<ShapeCheck("N,inChannels,H,W", ReturnShape="N,outChannels,1+(1+H/2),1+(1+W/2)")>]
    override _.forward(input) =
        let res = (input, blocks) ||> Array.fold (fun last layer -> layer.forward last) 
        res

type Kind =
    | EfficientnetB0
    | EfficientnetB1
    | EfficientnetB2
    | EfficientnetB3
    | EfficientnetB4
    | EfficientnetB5
    | EfficientnetB6
    | EfficientnetB7
    | EfficientnetB8
    | EfficientnetL2

/// default settings are efficientnetB0 (baseline) network
/// resolution is here to show what the network can take as input, it doesn't set anything!
//[<ShapeCheck(15)>]
type EfficientNet(?classCount: int,
        ?width: double,
        ?depth: double,
        ?resolution: int,
        ?dropout: double) =
    inherit Model()
    let zeroPad = ZeroPadding2d(0,1)

    let classCount = defaultArg classCount 1000
    let width = defaultArg width 1.0
    let depth = defaultArg depth 1.0
    let resolution = defaultArg resolution 224
    let dropout = defaultArg dropout 0.2

    let inputConv = Conv2d(3, makeDivisible(32, width), kernelSize=3, stride=2 (* , padding="valid" *))
    let inputConvBatchNorm = BatchNorm2d(numFeatures=makeDivisible(32, width))

    let initialMBConv = InitialMBConvBlock(32, 16, width=width)

    let residualBlockStack1 = MBConvBlockStack(16, 24, width=width, blockCount=2, depth=depth)
    let residualBlockStack2 = MBConvBlockStack(24, 40, width=width, kernel=5, blockCount=2, depth=depth)
    let residualBlockStack3 = MBConvBlockStack(40, 80, width=width, blockCount=3, depth=depth)
    let residualBlockStack4 = MBConvBlockStack(80, 112, width=width, initialStride=1, kernel=5, blockCount=3, depth=depth)
    let residualBlockStack5 = MBConvBlockStack(112, 192, width=width, kernel=5, blockCount=4, depth=depth)
    let residualBlockStack6 = MBConvBlockStack(192, 320, width=width, initialStride=1, blockCount=1, depth=depth)

    let outputConv = Conv2d(makeDivisible(320, width), makeDivisible(1280, width), kernelSize=1,stride=1, padding=0 (* "same " *))
    let outputConvBatchNorm = BatchNorm2d(numFeatures=makeDivisible(1280, width))
    let avgPool = GlobalAvgPool2d()

    let dropoutProb = Dropout(dropout)
    let outputClassifier = Linear(inFeatures= makeDivisible(1280, width), outFeatures=classCount)

    [<ShapeCheck("N,3,H,W", ReturnShape="N,classCount")>]
    override _.forward(input) =
        let convolved = input |> zeroPad.forward |> inputConv .forward |> inputConvBatchNorm .forward |> dsharp.silu
        let initialBlock = convolved |> initialMBConv.forward
        let backbone = 
            initialBlock 
            |> residualBlockStack1.forward |> residualBlockStack2.forward 
            |> residualBlockStack3.forward |> residualBlockStack4.forward
            |> residualBlockStack5.forward |> residualBlockStack6.forward
        let output = backbone |> outputConv.forward |> outputConvBatchNorm.forward |> dsharp.silu
        output |> avgPool.forward |> dropoutProb.forward |> outputClassifier.forward

    static member Create (kind: Kind, ?classCount: int) = 
        let classCount = defaultArg classCount 1000
        match kind with
        | EfficientnetB0 ->
            EfficientNet(classCount=classCount, width=1.0, depth=1.0, resolution=224, dropout=0.2)
        | EfficientnetB1 -> 
            EfficientNet(classCount=classCount, width=1.0, depth=1.1, resolution=240, dropout=0.2)
        | EfficientnetB2 ->
            EfficientNet(classCount=classCount, width=1.1, depth=1.2, resolution=260, dropout=0.3)
        | EfficientnetB3 ->
            EfficientNet(classCount=classCount, width=1.2, depth=1.4, resolution=300, dropout=0.3)
        | EfficientnetB4 ->
            EfficientNet(classCount=classCount, width=1.4, depth=1.8, resolution=380, dropout=0.4)
        | EfficientnetB5 ->
            EfficientNet(classCount=classCount, width=1.6, depth=2.2, resolution=456, dropout=0.4)
        | EfficientnetB6 ->
            EfficientNet(classCount=classCount, width=1.8, depth=2.6, resolution=528, dropout=0.5)
        | EfficientnetB7 ->
            EfficientNet(classCount=classCount, width=2.0, depth=3.1, resolution=600, dropout=0.5)
        | EfficientnetB8 ->
            EfficientNet(classCount=classCount, width=2.2, depth=3.6, resolution=672, dropout=0.5)
        | EfficientnetL2 ->
            // https://arxiv.org/abs/1911.04252
            EfficientNet(classCount=classCount, width=4.3, depth=5.3, resolution=800, dropout=0.5)
