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

// Original Paper:
// "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
// Mingxing Tan, Quoc V. Le
// https://arxiv.org/abs/1905.11946
// Notes: Default baseline (B0) network, see table 1

/// some utility functions to help generate network variants
/// original: https://github.com/tensorflow/tpu/blob/d6f2ef3edfeb4b1c2039b81014dc5271a7753832/models/official/efficientnet/efficientnet_model.py#L138
fileprivate let resizeDepth(blockCount: int, depth: double) = Int {
    /// Multiply + round up the number of blocks based on depth multiplier
    let newFilterCount = depth * double(blockCount)
    newFilterCount.round(.up)
    return int(newFilterCount)


fileprivate let makeDivisible(filter: int, width: double, divisor: double = 8.0) = Int {
    /// Return a filter multiplied by width, rounded down and evenly divisible by the divisor
    let filterMult = double(filter) * width
    let filterAdd = double(filterMult) + (divisor / 2.0)
    let div = filterAdd / divisor
    div.round(.down)
    div = div * double(divisor)
    let newFilterCount = max(1, int(div))
    if newFilterCount < int(0.9 * double(filter)) = 
        newFilterCount <- newFilterCount + int(divisor)

    return int(newFilterCount)


fileprivate let roundFilterPair(filters: (Int, Int), width: double) = (Int, Int) = 
    return (
        makeDivisible(filter: filters.0, width: width),
        makeDivisible(filter: filters.1, width: width)
    )


type InitialMBConvBlock: Layer {
    let hiddenDimension: int
    let dConv: DepthwiseConv2D<Float>
    let batchNormDConv: BatchNorm<Float>
    let seAveragePool = GlobalAvgPool2D<Float>()
    let seReduceConv: Conv2D<Float>
    let seExpandConv: Conv2D<Float>
    let conv2: Conv2D<Float>
    let batchNormConv2: BatchNorm<Float>

    init(filters: (Int, Int), width: double) = 
        let filterMult = roundFilterPair(filters: filters, width: width)
        self.hiddenDimension = filterMult.0
        dConv = DepthwiseConv2D<Float>(
            filterShape=(3, 3, filterMult.0, 1),
            stride=1,
            padding="same")
        seReduceConv = Conv2d(
            filterShape=(1, 1, filterMult.0, makeDivisible(filter: 8, width: width)),
            stride=1,
            padding="same")
        seExpandConv = Conv2d(
            filterShape=(1, 1, makeDivisible(filter: 8, width: width), filterMult.0),
            stride=1,
            padding="same")
        conv2 = Conv2d(
            filterShape=(1, 1, filterMult.0, filterMult.1),
            stride=1,
            padding="same")
        batchNormDConv = BatchNorm(featureCount=filterMult.0)
        batchNormConv2 = BatchNorm(featureCount=filterMult.1)


    
    override _.forward(input) =
        let depthwise = swish(batchNormDConv(dConv(input)))
        let seAvgPoolReshaped = seAveragePool(depthwise).reshape([
            input.shape.[0], 1, 1, self.hiddenDimension
        ])
        let squeezeExcite = depthwise
            * sigmoid(seExpandConv(swish(seReduceConv(seAvgPoolReshaped))))
        return batchNormConv2(conv2(squeezeExcite))



type MBConvBlock: Layer {
    let addResLayer: bool
    let strides = [Int, Int)
    let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    let hiddenDimension: int

    let conv1: Conv2D<Float>
    let batchNormConv1: BatchNorm<Float>
    let dConv: DepthwiseConv2D<Float>
    let batchNormDConv: BatchNorm<Float>
    let seAveragePool = GlobalAvgPool2D<Float>()
    let seReduceConv: Conv2D<Float>
    let seExpandConv: Conv2D<Float>
    let conv2: Conv2D<Float>
    let batchNormConv2: BatchNorm<Float>

    init(
        filters: (Int, Int),
        width: double,
        depthMultiplier: int = 6,
        strides = [Int, Int) = (1, 1),
        kernel: (Int, Int) = (3, 3)
    ) = 
        self.strides = strides
        self.addResLayer = filters.0 = filters.1 && strides = (1, 1)

        let filterMult = roundFilterPair(filters: filters, width: width)
        self.hiddenDimension = filterMult.0 * depthMultiplier
        let reducedDimension = max(1, int(filterMult.0 / 4))
        conv1 = Conv2d(
            filterShape=(1, 1, filterMult.0, hiddenDimension),
            stride=1,
            padding="same")
        dConv = DepthwiseConv2D<Float>(
            filterShape=(kernel.0, kernel.1, hiddenDimension, 1),
            strides: strides,
            padding: strides = (1, 1) ? .same : .valid)
        seReduceConv = Conv2d(
            filterShape=(1, 1, hiddenDimension, reducedDimension),
            stride=1,
            padding="same")
        seExpandConv = Conv2d(
            filterShape=(1, 1, reducedDimension, hiddenDimension),
            stride=1,
            padding="same")
        conv2 = Conv2d(
            filterShape=(1, 1, hiddenDimension, filterMult.1),
            stride=1,
            padding="same")
        batchNormConv1 = BatchNorm(featureCount=hiddenDimension)
        batchNormDConv = BatchNorm(featureCount=hiddenDimension)
        batchNormConv2 = BatchNorm(featureCount=filterMult.1)


    
    override _.forward(input) =
        let piecewise = swish(batchNormConv1(conv1(input)))
        let depthwise: Tensor
        if self.strides = (1, 1) = 
            depthwise = swish(batchNormDConv(dConv(piecewise)))
        else
            depthwise = swish(batchNormDConv(dConv(zeroPad(piecewise))))

        let seAvgPoolReshaped = seAveragePool(depthwise).reshape([
            input.shape.[0], 1, 1, self.hiddenDimension
        ])
        let squeezeExcite = depthwise
            * sigmoid(seExpandConv(swish(seReduceConv(seAvgPoolReshaped))))
        let piecewiseLinear = batchNormConv2(conv2(squeezeExcite))

        if self.addResLayer then
            return input + piecewiseLinear
        else
            return piecewiseLinear




type MBConvBlockStack: Layer {
    let blocks: [MBConvBlock] = []

    init(
        filters: (Int, Int),
        width: double,
        initialStrides: (Int, Int) = (2, 2),
        kernel: (Int, Int) = (3, 3),
        blockCount: int,
        depth: double
    ) = 
        let blockMult = resizeDepth(blockCount: blockCount, depth: depth)
        self.blocks = [
            MBConvBlock(
                filters: (filters.0, filters.1), width: width,
                strides: initialStrides, kernel: kernel)
        ]
        for _ in 1..<blockMult {
            self.blocks.append(
                MBConvBlock(
                    filters: (filters.1, filters.1),
                    width: width, kernel: kernel))



    
    override _.forward(input) =
        return blocks.differentiableReduce(input) =  $1($0)



type EfficientNet: Layer {
    let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    let inputConv: Conv2D<Float>
    let inputConvBatchNorm: BatchNorm<Float>
    let initialMBConv: InitialMBConvBlock

    let residualBlockStack1: MBConvBlockStack
    let residualBlockStack2: MBConvBlockStack
    let residualBlockStack3: MBConvBlockStack
    let residualBlockStack4: MBConvBlockStack
    let residualBlockStack5: MBConvBlockStack
    let residualBlockStack6: MBConvBlockStack

    let outputConv: Conv2D<Float>
    let outputConvBatchNorm: BatchNorm<Float>
    let avgPool = GlobalAvgPool2D<Float>()
    let dropoutProb: Dropout<Float>
    let outputClassifier: Dense

    /// default settings are efficientnetB0 (baseline) network
    /// resolution is here to show what the network can take as input, it doesn't set anything!
    public init(
        classCount: int = 1000,
        width: double = 1.0,
        depth: double = 1.0,
        resolution: int = 224,
        dropout: Double = 0.2
    ) = 
        inputConv = Conv2d(
            filterShape=(3, 3, 3, makeDivisible(filter: 32, width: width)),
            stride=2,
            padding: .valid)
        inputConvBatchNorm = BatchNorm(featureCount=makeDivisible(filter: 32, width: width))

        initialMBConv = InitialMBConvBlock(filters: (32, 16), width: width)

        residualBlockStack1 = MBConvBlockStack(
            filters: (16, 24), width: width,
            blockCount: 2, depth: depth)
        residualBlockStack2 = MBConvBlockStack(
            filters: (24, 40), width: width,
            kernel: (5, 5), blockCount: 2, depth: depth)
        residualBlockStack3 = MBConvBlockStack(
            filters: (40, 80), width: width,
            blockCount: 3, depth: depth)
        residualBlockStack4 = MBConvBlockStack(
            filters: (80, 112), width: width,
            initialStrides: (1, 1), kernel: (5, 5), blockCount: 3, depth: depth)
        residualBlockStack5 = MBConvBlockStack(
            filters: (112, 192), width: width,
            kernel: (5, 5), blockCount: 4, depth: depth)
        residualBlockStack6 = MBConvBlockStack(
            filters: (192, 320), width: width,
            initialStrides: (1, 1), blockCount: 1, depth: depth)

        outputConv = Conv2d(
            filterShape=(
                1, 1,
                makeDivisible(filter: 320, width: width), makeDivisible(filter: 1280, width: width)
            ),
            stride=1,
            padding="same")
        outputConvBatchNorm = BatchNorm(featureCount=makeDivisible(filter: 1280, width: width))

        dropoutProb = Dropout(probability: dropout)
        outputClassifier = Dense(
            inputSize= makeDivisible(filter: 1280, width: width),
            outputSize=classCount)


    
    override _.forward(input) =
        let convolved = swish(input |> zeroPad, inputConv, inputConvBatchNorm))
        let initialBlock = initialMBConv(convolved)
        let backbone = initialBlock.sequenced(
            through: residualBlockStack1, residualBlockStack2,
            residualBlockStack3, residualBlockStack4, residualBlockStack5, residualBlockStack6)
        let output = swish(backbone |> outputConv, outputConvBatchNorm))
        return output |> avgPool, dropoutProb, outputClassifier)



extension EfficientNet {
    type Kind {
        case efficientnetB0
        case efficientnetB1
        case efficientnetB2
        case efficientnetB3
        case efficientnetB4
        case efficientnetB5
        case efficientnetB6
        case efficientnetB7
        case efficientnetB8
        case efficientnetL2


    public init(kind: Kind, classCount: int = 1000) = 
        match kind with
        | .efficientnetB0 ->
            self.init(classCount: classCount, width: 1.0, depth: 1.0, resolution: 224, dropout: 0.2)
        | .efficientnetB1 ->
            self.init(classCount: classCount, width: 1.0, depth: 1.1, resolution: 240, dropout: 0.2)
        | .efficientnetB2 ->
            self.init(classCount: classCount, width: 1.1, depth: 1.2, resolution: 260, dropout: 0.3)
        | .efficientnetB3 ->
            self.init(classCount: classCount, width: 1.2, depth: 1.4, resolution: 300, dropout: 0.3)
        | .efficientnetB4 ->
            self.init(classCount: classCount, width: 1.4, depth: 1.8, resolution: 380, dropout: 0.4)
        | .efficientnetB5 ->
            self.init(classCount: classCount, width: 1.6, depth: 2.2, resolution: 456, dropout: 0.4)
        | .efficientnetB6 ->
            self.init(classCount: classCount, width: 1.8, depth: 2.6, resolution: 528, dropout: 0.5)
        | .efficientnetB7 ->
            self.init(classCount: classCount, width: 2.0, depth: 3.1, resolution: 600, dropout: 0.5)
        | .efficientnetB8 ->
            self.init(classCount: classCount, width: 2.2, depth: 3.6, resolution: 672, dropout: 0.5)
        | .efficientnetL2 ->
            // https://arxiv.org/abs/1911.04252
            self.init(classCount: classCount, width: 4.3, depth: 5.3, resolution: 800, dropout: 0.5)



