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

module Models.ImageClassification.MobileNetV2

open DiffSharp
open DiffSharp.Model

// Original Paper:
// "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
// Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
// https://arxiv.org/abs/1801.04381


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

let roundFilterPair((f1, f2), width: double) =
    makeDivisible(f1, width),makeDivisible(f2, width)

type InitialInvertedBottleneckBlock(filters: (int * int), widthMultiplier: double) =
    inherit Model()
    let filterMult0, filterMult1 = roundFilterPair(filters, widthMultiplier)
    let dConv = DepthwiseConv2d(filterMult0, 1, kernelSize=3, stride=1, padding=1 (* "same " *))
    let conv2 = Conv2d(filterMult0, filterMult1, kernelSize=1, stride=1, padding=0 (* "same " *))
    let batchNormDConv = BatchNorm2d(numFeatures=filterMult0)
    let batchNormConv = BatchNorm2d(numFeatures=filterMult1)
    
    override _.forward(input) =
        let depthwise = dsharp.relu6(batchNormDConv.forward(dConv.forward(input)))
        batchNormConv.forward(conv2.forward(depthwise))

type InvertedBottleneckBlock(filters: (int * int),
        widthMultiplier: double,
        ?depthMultiplier: int,
        ?strides) =
    inherit Model()
    let filters0, filters1 = filters
    let depthMultiplier = defaultArg depthMultiplier 6
    let stride0, stride1 = defaultArg strides (1,1)
    let zeroPad = ZeroPadding2d(0,1)

    let addResLayer = filters0 = filters1 && (stride0, stride1) = (1, 1)

    let filterMult0, filterMult1 = roundFilterPair(filters, widthMultiplier)
    let hiddenDimension = filterMult0 * depthMultiplier
    let conv1 = Conv2d(filterMult0, hiddenDimension, kernelSize=1, stride=1, padding=0 (* "same " *))
    let dConv = DepthwiseConv2d(hiddenDimension, 1, kernelSize=3, strides=[stride0; stride1] (* , padding: strides = (1, 1) ? .same : .valid *) )
    let conv2 = Conv2d(hiddenDimension, filterMult1, kernelSize=1, stride=1, padding=0 (* "same " *))
    let batchNormConv1 = BatchNorm2d(numFeatures=hiddenDimension)
    let batchNormDConv = BatchNorm2d(numFeatures=hiddenDimension)
    let batchNormConv2 = BatchNorm2d(numFeatures=filterMult1)
    
    override _.forward(input) =
        let pointwise = dsharp.relu6(batchNormConv1.forward(conv1.forward(input)))
        let depthwise =
            if (stride0, stride1) = (1, 1) then
                dsharp.relu6(batchNormDConv.forward(dConv.forward(pointwise)))
            else
                dsharp.relu6(batchNormDConv.forward(dConv.forward(zeroPad.forward(pointwise))))

        let pointwiseLinear = batchNormConv2.forward(conv2.forward(depthwise))

        if addResLayer then
            input + pointwiseLinear
        else
            pointwiseLinear

type InvertedBottleneckBlockStack(filters: (int * int),
        widthMultiplier: double,
        blockCount: int,
        ?initialStrides) =
    inherit Model()

    let initialStrides = defaultArg initialStrides (2,2)
    let blocks = 
        [| InvertedBottleneckBlock(filters=filters, widthMultiplier=widthMultiplier,strides=initialStrides)
           for _ in 1..blockCount-1 do
               InvertedBottleneckBlock(filters=(snd filters, snd filters), widthMultiplier=widthMultiplier)
        |]
    
    override _.forward(input) =
        (input, blocks) ||> Array.fold (fun last layer -> layer.forward last) 

type MobileNetV2(?classCount: int, ?widthMultiplier: double) =
    inherit Model()
    let classCount = defaultArg classCount 1000
    let widthMultiplier = defaultArg widthMultiplier 1.0
    let zeroPad = ZeroPadding2d(0,1)
    let avgPool = GlobalAvgPool2d()

    let inputConv = Conv2d(3, makeDivisible(32, widthMultiplier), kernelSize=3, stride=2 (* , padding="valid" *))
    let inputConvBatchNorm = BatchNorm2d(numFeatures= makeDivisible(32, widthMultiplier))

    let initialInvertedBottleneck = InitialInvertedBottleneckBlock(filters=(32, 16), widthMultiplier=widthMultiplier)
    let residualBlockStack1 = InvertedBottleneckBlockStack(filters=(16, 24), widthMultiplier=widthMultiplier, blockCount=2)
    let residualBlockStack2 = InvertedBottleneckBlockStack(filters=(24, 32), widthMultiplier=widthMultiplier, blockCount=3)
    let residualBlockStack3 = InvertedBottleneckBlockStack(filters=(32, 64), widthMultiplier=widthMultiplier, blockCount=4)
    let residualBlockStack4 = InvertedBottleneckBlockStack(filters=(64, 96), widthMultiplier=widthMultiplier, blockCount=3, initialStrides=(1, 1))
    let residualBlockStack5 = InvertedBottleneckBlockStack(filters=(96, 160), widthMultiplier=widthMultiplier, blockCount=3)

    let invertedBottleneckBlock16 = InvertedBottleneckBlock(filters=(160, 320), widthMultiplier=widthMultiplier)

    let lastBlockChannels = makeDivisible(1280, widthMultiplier)
    let lastBlockChannels = 
        if widthMultiplier < 1.0 then
            // paper: "One minor implementation difference, with [arxiv:1704.04861] is that for
            // multipliers less than one, we apply width multiplier to all layers except the very
            // last convolutional layer."
            1280
        else
            lastBlockChannels

    let outputConv = Conv2d(makeDivisible(320, widthMultiplier), lastBlockChannels, kernelSize=1, stride=1, padding=0 (* "same " *))
    let outputConvBatchNorm = BatchNorm2d(numFeatures=lastBlockChannels)

    let outputClassifier = Linear(lastBlockChannels, classCount)

    override _.forward(input) =
        let convolved = dsharp.relu6(input |> zeroPad.forward |> inputConv.forward |> inputConvBatchNorm.forward)
        let initialConv = initialInvertedBottleneck.forward(convolved)
        let backbone = initialConv.sequenced(residualBlockStack1, residualBlockStack2, residualBlockStack3, residualBlockStack4, residualBlockStack5)
        let output = dsharp.relu6(outputConvBatchNorm.forward((outputConv.forward(invertedBottleneckBlock16.forward(backbone)))))
        output |> avgPool.forward |> outputClassifier.forward


