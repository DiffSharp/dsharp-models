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

module Models.ImageClassification.MobileNetV3

open DiffSharp
open DiffSharp.Model

// Original Paper: "Searching for MobileNetV3"
// Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang,
// Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
// https://arxiv.org/abs/1905.02244

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

type ActivationType =
    | HardSwish
    | Relu

type SqueezeExcitationBlock(inputOutputSize: int, reducedSize: int) =
    inherit Model()
    // https://arxiv.org/abs/1709.01507
    let averagePool = GlobalAvgPool2d()

    let inputOutputSize = inputOutputSize
    let reduceConv = Conv2d(inputOutputSize, reducedSize, kernelSize=1, stride=1, padding=0 (* "same " *))
    let expandConv = Conv2d(reducedSize, inputOutputSize, kernelSize=1, stride=1, padding=0 (* "same " *))
    
    override _.forward(input) =
        let avgPoolReshaped = averagePool.forward(input).view([ input.shape.[0]; 1; 1; inputOutputSize ])
        input * dsharp.hardsigmoid(expandConv.forward(dsharp.relu(reduceConv.forward(avgPoolReshaped))))

type InitialInvertedResidualBlock(filters: (int * int),
        widthMultiplier: double,
        ?stride: int,
        ?kernelSize: int,
        ?useSELayer: bool,
        ?activation) =
    inherit Model()
    let stride = defaultArg stride 1
    let kernelSize = defaultArg kernelSize 3
    let useSELayer = defaultArg useSELayer false
    let activation = defaultArg activation Relu
    let filters0, filters1 = filters
    let addResLayer = filters0 = filters1 && stride = 1

    let filterMult0, filterMult1 = roundFilterPair(filters, widthMultiplier)
    let hiddenDimension = filterMult0 * 1
    let reducedDimension = hiddenDimension / 4

    let dConv = DepthwiseConv2d(filterMult0, 1, kernelSize=kernelSize, stride=1, padding=0 (* "same " *))
    let seBlock = SqueezeExcitationBlock(inputOutputSize=hiddenDimension, reducedSize=reducedDimension)
    let conv2 = Conv2d(hiddenDimension, filterMult1, kernelSize=1, stride=1, padding=0 (* "same " *))
    let batchNormDConv = BatchNorm2d(numFeatures=filterMult0)
    let batchNormConv2 = BatchNorm2d(numFeatures=filterMult1)
    
    override _.forward(input) =
        let depthwise = batchNormDConv.forward(dConv.forward(input))
        let depthwise = 
            match activation with
            | HardSwish -> dsharp.hardswish(depthwise)
            | Relu -> dsharp.relu(depthwise)

        let squeezeExcite =
            if useSELayer then
                seBlock.forward(depthwise)
            else
                depthwise

        let piecewiseLinear = batchNormConv2.forward(conv2.forward(squeezeExcite))

        if addResLayer then
            input + piecewiseLinear
        else
            piecewiseLinear

type InvertedResidualBlock(filters: (int * int),
        widthMultiplier: double,
        expansionFactor: double,
        ?stride: int,
        ?kernelSize: int,
        ?useSELayer: bool,
        ?activation) =
    inherit Model()
    let stride = defaultArg stride 1
    let kernelSize = defaultArg kernelSize 3
    let useSELayer = defaultArg useSELayer false
    let activation = defaultArg activation Relu
    let filters0, filters1 = filters
    let zeroPad = ZeroPadding2d(((0, 1), (0, 1)))

    let addResLayer = filters0 = filters1 && stride = 1

    let filterMult0, filterMult1 = roundFilterPair(filters, widthMultiplier)
    let hiddenDimension = int(double(filterMult0) * expansionFactor)
    let reducedDimension = hiddenDimension / 4

    let conv1 = Conv2d(filterMult0, hiddenDimension, kernelSize=1, stride=1, padding=kernelSize/2 (* "same " *))
    let dConv = DepthwiseConv2d(hiddenDimension, 1, kernelSize=kernelSize, stride=stride (* , padding: strides = (1, 1) ? .same : .valid *) )
    let seBlock = SqueezeExcitationBlock(inputOutputSize=hiddenDimension, reducedSize=reducedDimension)
    let conv2 = Conv2d(hiddenDimension, filterMult1, kernelSize=1, stride=1, padding=kernelSize/2 (* "same " *))
    let batchNormConv1 = BatchNorm2d(numFeatures=hiddenDimension)
    let batchNormDConv = BatchNorm2d(numFeatures=hiddenDimension)
    let batchNormConv2 = BatchNorm2d(numFeatures=filterMult1)

    override _.forward(input) =
        let piecewise = batchNormConv1.forward(conv1.forward(input))
        let piecewise = 
            match activation with
            | HardSwish -> dsharp.hardswish(piecewise)
            | Relu -> dsharp.relu(piecewise)

        let depthwise =
            if stride = 1 then
                batchNormDConv.forward(dConv.forward(piecewise))
            else
                batchNormDConv.forward(dConv.forward(zeroPad.forward(piecewise)))

        let depthwise =
            match activation with
            | HardSwish -> dsharp.hardswish(depthwise)
            | Relu -> dsharp.relu(depthwise)

        let squeezeExcite =
            if useSELayer then
                seBlock.forward(depthwise)
            else
                depthwise

        let piecewiseLinear = batchNormConv2.forward(conv2.forward(squeezeExcite))

        if addResLayer then
            input + piecewiseLinear
        else
            piecewiseLinear

type MobileNetV3Large(?classCount: int, ?widthMultiplier: double, ?dropout: double) =
    inherit Model()
    let classCount = defaultArg classCount 1000
    let widthMultiplier = defaultArg widthMultiplier 1.0
    let dropout = defaultArg dropout 0.2
    let zeroPad = ZeroPadding2d(((0, 1), (0, 1)))

    let avgPool = GlobalAvgPool2d()
    let flatten = Flatten()

    let inputConv = Conv2d(3, makeDivisible(16, widthMultiplier), kernelSize=3, stride=2, padding=1 (* "same " *))
    let inputConvBatchNorm = BatchNorm2d(makeDivisible(16, widthMultiplier))

    let invertedResidualBlock1 = InitialInvertedResidualBlock(filters=(16, 16), widthMultiplier=widthMultiplier)
    let invertedResidualBlock2 = InvertedResidualBlock(filters=(16, 24), widthMultiplier=widthMultiplier, expansionFactor=4.0, stride=2)
    let invertedResidualBlock3 = InvertedResidualBlock(filters=(24, 24), widthMultiplier=widthMultiplier, expansionFactor=3.0)
    let invertedResidualBlock4 = InvertedResidualBlock(filters=(24, 40), widthMultiplier=widthMultiplier, expansionFactor=3.0, stride=2, kernelSize=5, useSELayer=true)
    let invertedResidualBlock5 = InvertedResidualBlock(filters=(40, 40), widthMultiplier=widthMultiplier, expansionFactor=3.0, kernelSize=5, useSELayer=true)
    let invertedResidualBlock6 = InvertedResidualBlock(filters=(40, 40), widthMultiplier=widthMultiplier, expansionFactor=3.0, kernelSize=5, useSELayer=true)
    let invertedResidualBlock7 = InvertedResidualBlock(filters=(40, 80), widthMultiplier=widthMultiplier, expansionFactor=6.0, stride=2, activation= HardSwish)
    let invertedResidualBlock8 = InvertedResidualBlock(filters=(80, 80), widthMultiplier=widthMultiplier, expansionFactor=2.5, activation= HardSwish)
    let invertedResidualBlock9 = InvertedResidualBlock(filters=(80, 80), widthMultiplier=widthMultiplier, expansionFactor=184.0 / 80.0, activation= HardSwish)
    let invertedResidualBlock10 = InvertedResidualBlock(filters=(80, 80), widthMultiplier=widthMultiplier, expansionFactor=184.0 / 80.0, activation= HardSwish)
    let invertedResidualBlock11 = InvertedResidualBlock(filters=(80, 112), widthMultiplier=widthMultiplier, expansionFactor=6.0, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock12 = InvertedResidualBlock(filters=(112, 112), widthMultiplier=widthMultiplier, expansionFactor=6.0, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock13 = InvertedResidualBlock(filters=(112, 160), widthMultiplier=widthMultiplier, expansionFactor=6.0, stride=2, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock14 = InvertedResidualBlock(filters=(160, 160), widthMultiplier=widthMultiplier, expansionFactor=6.0, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock15 = InvertedResidualBlock(filters=(160, 160), widthMultiplier=widthMultiplier, expansionFactor=6.0, kernelSize=5, useSELayer=true, activation= HardSwish)

    let lastConvChannel = makeDivisible(960, widthMultiplier)
    let outputConv = Conv2d(makeDivisible(160, widthMultiplier), lastConvChannel, kernelSize=1, stride=1, padding=0 (* "same " *))
    let outputConvBatchNorm = BatchNorm2d(numFeatures=lastConvChannel)

    let lastPointChannel = if widthMultiplier > 1.0 then makeDivisible(1280, widthMultiplier) else 1280
    let finalConv = Conv2d(lastConvChannel, lastPointChannel, kernelSize=1, stride=1, padding=0 (* "same " *))
    let dropoutLayer = Dropout(dropout)
    let classiferConv = Conv2d(lastPointChannel, classCount, kernelSize=1, stride=1, padding=0 (* "same " *))
    
    override _.forward(input) =
        let initialConv = dsharp.hardswish(input |> zeroPad.forward |> inputConv.forward |> inputConvBatchNorm.forward)
        let backbone1 = initialConv.sequenced(invertedResidualBlock1, invertedResidualBlock2, invertedResidualBlock3, invertedResidualBlock4, invertedResidualBlock5)
        let backbone2 = backbone1.sequenced(invertedResidualBlock6, invertedResidualBlock7, invertedResidualBlock8, invertedResidualBlock9, invertedResidualBlock10)
        let backbone3 = backbone2.sequenced(invertedResidualBlock11, invertedResidualBlock12, invertedResidualBlock13, invertedResidualBlock14, invertedResidualBlock15)
        let outputConvResult = dsharp.hardswish(outputConvBatchNorm.forward(outputConv.forward(backbone3)))
        let averagePool = avgPool.forward(outputConvResult).view([ input.shape.[0]; 1; 1; lastConvChannel ])
        let finalConvResult = dropoutLayer.forward(dsharp.hardswish(finalConv.forward(averagePool)))
        dsharp.flatten(classiferConv.forward(finalConvResult))

type MobileNetV3Small(?classCount: int, ?widthMultiplier: double, ?dropout: double) =
    inherit Model()
    let classCount = defaultArg classCount 10000
    let widthMultiplier = defaultArg widthMultiplier 1.0
    let dropout = defaultArg dropout 0.2
    let zeroPad = ZeroPadding2d(((0, 1), (0, 1)))

    let avgPool = GlobalAvgPool2d()
    let flatten = Flatten()

    let inputConv = Conv2d(3, makeDivisible(16, widthMultiplier), kernelSize=3, stride=2, padding=1 (* "same " *))
    let inputConvBatchNorm = BatchNorm2d(makeDivisible(16, widthMultiplier))

    let invertedResidualBlock1 = InitialInvertedResidualBlock(filters=(16, 16), widthMultiplier=widthMultiplier, stride=2, useSELayer=true)
    let invertedResidualBlock2 = InvertedResidualBlock(filters=(16, 24), widthMultiplier=widthMultiplier, expansionFactor=72.0 / 16.0, stride=2)
    let invertedResidualBlock3 = InvertedResidualBlock(filters=(24, 24), widthMultiplier=widthMultiplier, expansionFactor=88.0 / 24.0)
    let invertedResidualBlock4 = InvertedResidualBlock(filters=(24, 40), widthMultiplier=widthMultiplier, expansionFactor=4.0, stride=2, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock5 = InvertedResidualBlock(filters=(40, 40), widthMultiplier=widthMultiplier, expansionFactor=6.0, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock6 = InvertedResidualBlock(filters=(40, 40), widthMultiplier=widthMultiplier, expansionFactor=6.0, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock7 = InvertedResidualBlock(filters=(40, 48), widthMultiplier=widthMultiplier, expansionFactor=3.0, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock8 = InvertedResidualBlock(filters=(48, 48), widthMultiplier=widthMultiplier, expansionFactor=3.0, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock9 = InvertedResidualBlock(filters=(48, 96), widthMultiplier=widthMultiplier, expansionFactor=6.0, stride=2, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock10 = InvertedResidualBlock(filters=(96, 96), widthMultiplier=widthMultiplier, expansionFactor=6.0, kernelSize=5, useSELayer=true, activation= HardSwish)
    let invertedResidualBlock11 = InvertedResidualBlock(filters=(96, 96), widthMultiplier=widthMultiplier, expansionFactor=6.0, kernelSize=5, useSELayer=true, activation= HardSwish)

    let lastConvChannel = makeDivisible(576, widthMultiplier)
    let outputConv = Conv2d(makeDivisible(96, widthMultiplier), lastConvChannel, kernelSize=1, stride=1, padding=0 (* "same " *))
    let outputConvBatchNorm = BatchNorm2d(numFeatures=lastConvChannel)

    let lastPointChannel =
        if widthMultiplier > 1.0 then makeDivisible(1280, widthMultiplier) else 1280
    let finalConv = Conv2d(lastConvChannel, lastPointChannel, kernelSize=1, stride=1, padding=0 (* "same " *))
    let dropoutLayer = Dropout(dropout)
    let classiferConv = Conv2d(lastPointChannel, classCount, kernelSize=1, stride=1, padding=0 (* "same " *))

    override _.forward(input) =
        let initialConv = dsharp.hardswish(input |> zeroPad.forward |> inputConv.forward |> inputConvBatchNorm.forward)
        let backbone1 = initialConv.sequenced(invertedResidualBlock1, invertedResidualBlock2, invertedResidualBlock3, invertedResidualBlock4, invertedResidualBlock5)
        let backbone2 = backbone1.sequenced(invertedResidualBlock6, invertedResidualBlock7, invertedResidualBlock8, invertedResidualBlock9, invertedResidualBlock10, invertedResidualBlock11)
        let outputConvResult = dsharp.hardswish(outputConvBatchNorm.forward(outputConv.forward(backbone2)))
        let averagePool = avgPool.forward(outputConvResult).view([ input.shape.[0]; 1; 1; lastConvChannel ])
        let finalConvResult = dropoutLayer.forward(dsharp.hardswish(finalConv.forward(averagePool)))
        dsharp.flatten(classiferConv.forward(finalConvResult))
