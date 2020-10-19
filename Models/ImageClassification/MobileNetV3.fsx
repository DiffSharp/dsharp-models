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

open DiffSharp

// Original Paper: "Searching for MobileNetV3"
// Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang,
// Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
// https://arxiv.org/abs/1905.02244

fileprivate let makeDivisible(filter: int, widthMultiplier: double = 1.0, divisor: double = 8.0)
    -> Int
{
    /// Return a filter multiplied by width, evenly divisible by the divisor
    let filterMult = double(filter) * widthMultiplier
    let filterAdd = double(filterMult) + (divisor / 2.0)
    let div = filterAdd / divisor
    div.round(.down)
    div = div * double(divisor)
    let newFilterCount = max(1, int(div))
    if newFilterCount < int(0.9 * double(filter)) = 
        newFilterCount <- newFilterCount + int(divisor)

    return int(newFilterCount)


fileprivate let roundFilterPair(filters: (Int, Int), widthMultiplier: double) = (Int, Int) = 
    return (
        makeDivisible(filter: filters.0, widthMultiplier: widthMultiplier),
        makeDivisible(filter: filters.1, widthMultiplier: widthMultiplier)
    )


type ActivationType {
    case hardSwish
    case relu


type SqueezeExcitationBlock: Layer {
    // https://arxiv.org/abs/1709.01507
    let averagePool = GlobalAvgPool2D<Float>()
    let reduceConv: Conv2D<Float>
    let expandConv: Conv2D<Float>
    @noDerivative let inputOutputSize: int

    public init(inputOutputSize: int, reducedSize: int) = 
        self.inputOutputSize = inputOutputSize
        reduceConv = Conv2d(
            filterShape=(1, 1, inputOutputSize, reducedSize),
            stride=1,
            padding="same")
        expandConv = Conv2d(
            filterShape=(1, 1, reducedSize, inputOutputSize),
            stride=1,
            padding="same")


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let avgPoolReshaped = averagePool(input).reshape([
            input.shape.[0], 1, 1, self.inputOutputSize,
        ])
        return input
            * hardSigmoid(expandConv(relu(reduceConv(avgPoolReshaped))))



type InitialInvertedResidualBlock: Layer {
    @noDerivative let addResLayer: bool
    @noDerivative let useSELayer: bool = false
    @noDerivative let activation= ActivationType = .relu

    let dConv: DepthwiseConv2D<Float>
    let batchNormDConv: BatchNorm<Float>
    let seBlock: SqueezeExcitationBlock
    let conv2: Conv2D<Float>
    let batchNormConv2: BatchNorm<Float>

    public init(
        filters: (Int, Int),
        widthMultiplier: double,
        strides = [Int, Int) = (1, 1),
        kernel: (Int, Int) = (3, 3),
        seLayer: bool = false,
        activation= ActivationType = .relu
    ) = 
        self.useSELayer = seLayer
        self.activation = activation
        self.addResLayer = filters.0 = filters.1 && strides = (1, 1)

        let filterMult = roundFilterPair(filters: filters, widthMultiplier: widthMultiplier)
        let hiddenDimension = filterMult.0 * 1
        let reducedDimension = hiddenDimension / 4

        dConv = DepthwiseConv2D<Float>(
            filterShape=(3, 3, filterMult.0, 1),
            stride=1,
            padding="same")
        seBlock = SqueezeExcitationBlock(
            inputOutputSize: hiddenDimension, reducedSize: reducedDimension)
        conv2 = Conv2d(
            filterShape=(1, 1, hiddenDimension, filterMult.1),
            stride=1,
            padding="same")
        batchNormDConv = BatchNorm(featureCount=filterMult.0)
        batchNormConv2 = BatchNorm(featureCount=filterMult.1)


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let depthwise = batchNormDConv(dConv(input))
        match self.activation {
        | .hardSwish -> depthwise = hardSwish(depthwise)
        | .relu -> depthwise = relu(depthwise)


        let squeezeExcite: Tensor
        if self.useSELayer then
            squeezeExcite = seBlock(depthwise)
        else
            squeezeExcite = depthwise


        let piecewiseLinear = batchNormConv2(conv2(squeezeExcite))

        if self.addResLayer then
            return input + piecewiseLinear
        else
            return piecewiseLinear




type InvertedResidualBlock: Layer {
    @noDerivative let strides = [Int, Int)
    @noDerivative let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    @noDerivative let addResLayer: bool
    @noDerivative let activation= ActivationType = .relu
    @noDerivative let useSELayer: bool

    let conv1: Conv2D<Float>
    let batchNormConv1: BatchNorm<Float>
    let dConv: DepthwiseConv2D<Float>
    let batchNormDConv: BatchNorm<Float>
    let seBlock: SqueezeExcitationBlock
    let conv2: Conv2D<Float>
    let batchNormConv2: BatchNorm<Float>

    public init(
        filters: (Int, Int),
        widthMultiplier: double,
        expansionFactor: double,
        strides = [Int, Int) = (1, 1),
        kernel: (Int, Int) = (3, 3),
        seLayer: bool = false,
        activation= ActivationType = .relu
    ) = 
        self.strides = strides
        self.addResLayer = filters.0 = filters.1 && strides = (1, 1)
        self.useSELayer = seLayer
        self.activation = activation

        let filterMult = roundFilterPair(filters: filters, widthMultiplier: widthMultiplier)
        let hiddenDimension = int(double(filterMult.0) * expansionFactor)
        let reducedDimension = hiddenDimension / 4

        conv1 = Conv2d(
            filterShape=(1, 1, filterMult.0, hiddenDimension),
            stride=1,
            padding="same")
        dConv = DepthwiseConv2D<Float>(
            filterShape=(kernel.0, kernel.1, hiddenDimension, 1),
            strides: strides,
            padding: strides = (1, 1) ? .same : .valid)
        seBlock = SqueezeExcitationBlock(
            inputOutputSize: hiddenDimension, reducedSize: reducedDimension)
        conv2 = Conv2d(
            filterShape=(1, 1, hiddenDimension, filterMult.1),
            stride=1,
            padding="same")
        batchNormConv1 = BatchNorm(featureCount=hiddenDimension)
        batchNormDConv = BatchNorm(featureCount=hiddenDimension)
        batchNormConv2 = BatchNorm(featureCount=filterMult.1)


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let piecewise = batchNormConv1(conv1(input))
        match self.activation {
        | .hardSwish -> piecewise = hardSwish(piecewise)
        | .relu -> piecewise = relu(piecewise)

        let depthwise: Tensor
        if self.strides = (1, 1) = 
            depthwise = batchNormDConv(dConv(piecewise))
        else
            depthwise = batchNormDConv(dConv(zeroPad(piecewise)))

        match self.activation {
        | .hardSwish -> depthwise = hardSwish(depthwise)
        | .relu -> depthwise = relu(depthwise)

        let squeezeExcite: Tensor
        if self.useSELayer then
            squeezeExcite = seBlock(depthwise)
        else
            squeezeExcite = depthwise


        let piecewiseLinear = batchNormConv2(conv2(squeezeExcite))

        if self.addResLayer then
            return input + piecewiseLinear
        else
            return piecewiseLinear




type MobileNetV3Large: Layer {
    @noDerivative let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    let inputConv: Conv2D<Float>
    let inputConvBatchNorm: BatchNorm<Float>

    let invertedResidualBlock1: InitialInvertedResidualBlock
    let invertedResidualBlock2: InvertedResidualBlock
    let invertedResidualBlock3: InvertedResidualBlock
    let invertedResidualBlock4: InvertedResidualBlock
    let invertedResidualBlock5: InvertedResidualBlock
    let invertedResidualBlock6: InvertedResidualBlock
    let invertedResidualBlock7: InvertedResidualBlock
    let invertedResidualBlock8: InvertedResidualBlock
    let invertedResidualBlock9: InvertedResidualBlock
    let invertedResidualBlock10: InvertedResidualBlock
    let invertedResidualBlock11: InvertedResidualBlock
    let invertedResidualBlock12: InvertedResidualBlock
    let invertedResidualBlock13: InvertedResidualBlock
    let invertedResidualBlock14: InvertedResidualBlock
    let invertedResidualBlock15: InvertedResidualBlock

    let outputConv: Conv2D<Float>
    let outputConvBatchNorm: BatchNorm<Float>

    let avgPool = GlobalAvgPool2D<Float>()
    let finalConv: Conv2D<Float>
    let dropoutLayer: Dropout<Float>
    let classiferConv: Conv2D<Float>
    let flatten = Flatten<Float>()

    @noDerivative let lastConvChannel: int

    public init(classCount: int = 1000, widthMultiplier: double = 1.0, dropout: Double = 0.2) = 
        inputConv = Conv2d(
            filterShape=(3, 3, 3, makeDivisible(filter: 16, widthMultiplier: widthMultiplier)),
            stride=2,
            padding="same")
        inputConvBatchNorm = BatchNorm(
            featureCount: makeDivisible(filter: 16, widthMultiplier: widthMultiplier))

        invertedResidualBlock1 = InitialInvertedResidualBlock(
            filters: (16, 16), widthMultiplier: widthMultiplier)
        invertedResidualBlock2 = InvertedResidualBlock(
            filters: (16, 24), widthMultiplier: widthMultiplier,
            expansionFactor: 4, stride=2)
        invertedResidualBlock3 = InvertedResidualBlock(
            filters: (24, 24), widthMultiplier: widthMultiplier,
            expansionFactor: 3)
        invertedResidualBlock4 = InvertedResidualBlock(
            filters: (24, 40), widthMultiplier: widthMultiplier,
            expansionFactor: 3, stride=2, kernel: (5, 5), seLayer: true)
        invertedResidualBlock5 = InvertedResidualBlock(
            filters: (40, 40), widthMultiplier: widthMultiplier,
            expansionFactor: 3, kernel: (5, 5), seLayer: true)
        invertedResidualBlock6 = InvertedResidualBlock(
            filters: (40, 40), widthMultiplier: widthMultiplier,
            expansionFactor: 3, kernel: (5, 5), seLayer: true)
        invertedResidualBlock7 = InvertedResidualBlock(
            filters: (40, 80), widthMultiplier: widthMultiplier,
            expansionFactor: 6, stride=2, activation= .hardSwish)
        invertedResidualBlock8 = InvertedResidualBlock(
            filters: (80, 80), widthMultiplier: widthMultiplier,
            expansionFactor: 2.5, activation= .hardSwish)
        invertedResidualBlock9 = InvertedResidualBlock(
            filters: (80, 80), widthMultiplier: widthMultiplier,
            expansionFactor: 184 / 80.0, activation= .hardSwish)
        invertedResidualBlock10 = InvertedResidualBlock(
            filters: (80, 80), widthMultiplier: widthMultiplier,
            expansionFactor: 184 / 80.0, activation= .hardSwish)
        invertedResidualBlock11 = InvertedResidualBlock(
            filters: (80, 112), widthMultiplier: widthMultiplier,
            expansionFactor: 6, seLayer: true, activation= .hardSwish)
        invertedResidualBlock12 = InvertedResidualBlock(
            filters: (112, 112), widthMultiplier: widthMultiplier,
            expansionFactor: 6, seLayer: true, activation= .hardSwish)
        invertedResidualBlock13 = InvertedResidualBlock(
            filters: (112, 160), widthMultiplier: widthMultiplier,
            expansionFactor: 6, stride=2, kernel: (5, 5), seLayer: true,
            activation= .hardSwish)
        invertedResidualBlock14 = InvertedResidualBlock(
            filters: (160, 160), widthMultiplier: widthMultiplier,
            expansionFactor: 6, kernel: (5, 5), seLayer: true, activation= .hardSwish)
        invertedResidualBlock15 = InvertedResidualBlock(
            filters: (160, 160), widthMultiplier: widthMultiplier,
            expansionFactor: 6, kernel: (5, 5), seLayer: true, activation= .hardSwish)

        lastConvChannel = makeDivisible(filter: 960, widthMultiplier: widthMultiplier)
        outputConv = Conv2d(
            filterShape=(
                1, 1, makeDivisible(filter: 160, widthMultiplier: widthMultiplier), lastConvChannel
            ),
            stride=1,
            padding="same")
        outputConvBatchNorm = BatchNorm(featureCount=lastConvChannel)

        let lastPointChannel =
            widthMultiplier > 1.0
            ? makeDivisible(filter: 1280, widthMultiplier: widthMultiplier) : 1280
        finalConv = Conv2d(
            filterShape=(1, 1, lastConvChannel, lastPointChannel),
            stride=1,
            padding="same")
        dropoutLayer = Dropout<Float>(probability: dropout)
        classiferConv = Conv2d(
            filterShape=(1, 1, lastPointChannel, classCount),
            stride=1,
            padding="same")


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let initialConv = hardSwish(
            input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
        let backbone1 = initialConv.sequenced(
            through: invertedResidualBlock1,
            invertedResidualBlock2, invertedResidualBlock3, invertedResidualBlock4,
            invertedResidualBlock5)
        let backbone2 = backbone1.sequenced(
            through: invertedResidualBlock6, invertedResidualBlock7,
            invertedResidualBlock8, invertedResidualBlock9, invertedResidualBlock10)
        let backbone3 = backbone2.sequenced(
            through: invertedResidualBlock11,
            invertedResidualBlock12, invertedResidualBlock13, invertedResidualBlock14,
            invertedResidualBlock15)
        let outputConvResult = hardSwish(outputConvBatchNorm(outputConv(backbone3)))
        let averagePool = avgPool(outputConvResult).reshape([
            input.shape.[0], 1, 1, self.lastConvChannel,
        ])
        let finalConvResult = dropoutLayer(hardSwish(finalConv(averagePool)))
        return flatten(classiferConv(finalConvResult))



type MobileNetV3Small: Layer {
    @noDerivative let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    let inputConv: Conv2D<Float>
    let inputConvBatchNorm: BatchNorm<Float>

    let invertedResidualBlock1: InitialInvertedResidualBlock
    let invertedResidualBlock2: InvertedResidualBlock
    let invertedResidualBlock3: InvertedResidualBlock
    let invertedResidualBlock4: InvertedResidualBlock
    let invertedResidualBlock5: InvertedResidualBlock
    let invertedResidualBlock6: InvertedResidualBlock
    let invertedResidualBlock7: InvertedResidualBlock
    let invertedResidualBlock8: InvertedResidualBlock
    let invertedResidualBlock9: InvertedResidualBlock
    let invertedResidualBlock10: InvertedResidualBlock
    let invertedResidualBlock11: InvertedResidualBlock

    let outputConv: Conv2D<Float>
    let outputConvBatchNorm: BatchNorm<Float>

    let avgPool = GlobalAvgPool2D<Float>()
    let finalConv: Conv2D<Float>
    let dropoutLayer: Dropout<Float>
    let classiferConv: Conv2D<Float>
    let flatten = Flatten<Float>()

    @noDerivative let lastConvChannel: int

    public init(classCount: int = 1000, widthMultiplier: double = 1.0, dropout: Double = 0.2) = 
        inputConv = Conv2d(
            filterShape=(3, 3, 3, makeDivisible(filter: 16, widthMultiplier: widthMultiplier)),
            stride=2,
            padding="same")
        inputConvBatchNorm = BatchNorm(
            featureCount: makeDivisible(filter: 16, widthMultiplier: widthMultiplier))

        invertedResidualBlock1 = InitialInvertedResidualBlock(
            filters: (16, 16), widthMultiplier: widthMultiplier,
            stride=2, seLayer: true)
        invertedResidualBlock2 = InvertedResidualBlock(
            filters: (16, 24), widthMultiplier: widthMultiplier,
            expansionFactor: 72.0 / 16.0, stride=2)
        invertedResidualBlock3 = InvertedResidualBlock(
            filters: (24, 24), widthMultiplier: widthMultiplier,
            expansionFactor: 88.0 / 24.0)
        invertedResidualBlock4 = InvertedResidualBlock(
            filters: (24, 40), widthMultiplier: widthMultiplier,
            expansionFactor: 4, stride=2, kernel: (5, 5), seLayer: true,
            activation= .hardSwish)
        invertedResidualBlock5 = InvertedResidualBlock(
            filters: (40, 40), widthMultiplier: widthMultiplier,
            expansionFactor: 6, kernel: (5, 5), seLayer: true, activation= .hardSwish)
        invertedResidualBlock6 = InvertedResidualBlock(
            filters: (40, 40), widthMultiplier: widthMultiplier,
            expansionFactor: 6, kernel: (5, 5), seLayer: true, activation= .hardSwish)
        invertedResidualBlock7 = InvertedResidualBlock(
            filters: (40, 48), widthMultiplier: widthMultiplier,
            expansionFactor: 3, kernel: (5, 5), seLayer: true, activation= .hardSwish)
        invertedResidualBlock8 = InvertedResidualBlock(
            filters: (48, 48), widthMultiplier: widthMultiplier,
            expansionFactor: 3, kernel: (5, 5), seLayer: true, activation= .hardSwish)
        invertedResidualBlock9 = InvertedResidualBlock(
            filters: (48, 96), widthMultiplier: widthMultiplier,
            expansionFactor: 6, stride=2, kernel: (5, 5), seLayer: true,
            activation= .hardSwish)
        invertedResidualBlock10 = InvertedResidualBlock(
            filters: (96, 96), widthMultiplier: widthMultiplier,
            expansionFactor: 6, kernel: (5, 5), seLayer: true, activation= .hardSwish)
        invertedResidualBlock11 = InvertedResidualBlock(
            filters: (96, 96), widthMultiplier: widthMultiplier,
            expansionFactor: 6, kernel: (5, 5), seLayer: true, activation= .hardSwish)

        lastConvChannel = makeDivisible(filter: 576, widthMultiplier: widthMultiplier)
        outputConv = Conv2d(
            filterShape=(
                1, 1, makeDivisible(filter: 96, widthMultiplier: widthMultiplier), lastConvChannel
            ),
            stride=1,
            padding="same")
        outputConvBatchNorm = BatchNorm(featureCount=lastConvChannel)

        let lastPointChannel =
            widthMultiplier > 1.0
            ? makeDivisible(filter: 1280, widthMultiplier: widthMultiplier) : 1280
        finalConv = Conv2d(
            filterShape=(1, 1, lastConvChannel, lastPointChannel),
            stride=1,
            padding="same")
        dropoutLayer = Dropout<Float>(probability: dropout)
        classiferConv = Conv2d(
            filterShape=(1, 1, lastPointChannel, classCount),
            stride=1,
            padding="same")


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let initialConv = hardSwish(
            input.sequenced(through: zeroPad, inputConv, inputConvBatchNorm))
        let backbone1 = initialConv.sequenced(
            through: invertedResidualBlock1,
            invertedResidualBlock2, invertedResidualBlock3, invertedResidualBlock4,
            invertedResidualBlock5)
        let backbone2 = backbone1.sequenced(
            through: invertedResidualBlock6, invertedResidualBlock7,
            invertedResidualBlock8, invertedResidualBlock9, invertedResidualBlock10,
            invertedResidualBlock11)
        let outputConvResult = hardSwish(outputConvBatchNorm(outputConv(backbone2)))
        let averagePool = avgPool(outputConvResult).reshape([
            input.shape.[0], 1, 1, lastConvChannel,
        ])
        let finalConvResult = dropoutLayer(hardSwish(finalConv(averagePool)))
        return flatten(classiferConv(finalConvResult))


