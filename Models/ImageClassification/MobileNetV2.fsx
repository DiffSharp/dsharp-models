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
// "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
// Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
// https://arxiv.org/abs/1801.04381

let makeDivisible(filter: int, widthMultiplier: double = 1.0, divisor: double = 8.0)
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


let roundFilterPair(filters: (Int, Int), widthMultiplier: double) = (Int, Int) = 
    return (
        makeDivisible(filter: filters.0, widthMultiplier: widthMultiplier),
        makeDivisible(filter: filters.1, widthMultiplier: widthMultiplier)
    )


type InitialInvertedBottleneckBlock: Layer {
    let dConv: DepthwiseConv2D<Float>
    let batchNormDConv: BatchNorm<Float>
    let conv2: Conv2D<Float>
    let batchNormConv: BatchNorm<Float>

    public init(filters: (Int, Int), widthMultiplier: double) = 
        let filterMult = roundFilterPair(filters: filters, widthMultiplier: widthMultiplier)
        dConv = DepthwiseConv2D<Float>(
            filterShape=(3, 3, filterMult.0, 1),
            stride=1,
            padding="same")
        conv2 = Conv2d(
            filterShape=(1, 1, filterMult.0, filterMult.1),
            stride=1,
            padding="same")
        batchNormDConv = BatchNorm(featureCount=filterMult.0)
        batchNormConv = BatchNorm(featureCount=filterMult.1)


    
    override _.forward(input) =
        let depthwise = relu6(batchNormDConv(dConv(input)))
        return batchNormConv(conv2(depthwise))



type InvertedBottleneckBlock: Layer {
    let addResLayer: bool
    let strides = [Int, Int)
    let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))

    let conv1: Conv2D<Float>
    let batchNormConv1: BatchNorm<Float>
    let dConv: DepthwiseConv2D<Float>
    let batchNormDConv: BatchNorm<Float>
    let conv2: Conv2D<Float>
    let batchNormConv2: BatchNorm<Float>

    public init(
        filters: (Int, Int),
        widthMultiplier: double,
        depthMultiplier: int = 6,
        strides = [Int, Int) = (1, 1)
    ) = 
        self.strides = strides
        self.addResLayer = filters.0 = filters.1 && strides = (1, 1)

        let filterMult = roundFilterPair(filters: filters, widthMultiplier: widthMultiplier)
        let hiddenDimension = filterMult.0 * depthMultiplier
        conv1 = Conv2d(
            filterShape=(1, 1, filterMult.0, hiddenDimension),
            stride=1,
            padding="same")
        dConv = DepthwiseConv2D<Float>(
            filterShape=(3, 3, hiddenDimension, 1),
            strides: strides,
            padding: strides = (1, 1) ? .same : .valid)
        conv2 = Conv2d(
            filterShape=(1, 1, hiddenDimension, filterMult.1),
            stride=1,
            padding="same")
        batchNormConv1 = BatchNorm(featureCount=hiddenDimension)
        batchNormDConv = BatchNorm(featureCount=hiddenDimension)
        batchNormConv2 = BatchNorm(featureCount=filterMult.1)


    
    override _.forward(input) =
        let pointwise = relu6(batchNormConv1(conv1(input)))
        let depthwise: Tensor
        if self.strides = (1, 1) = 
            depthwise = relu6(batchNormDConv(dConv(pointwise)))
        else
            depthwise = relu6(batchNormDConv(dConv(zeroPad(pointwise))))

        let pointwiseLinear = batchNormConv2(conv2(depthwise))

        if self.addResLayer then
            return input + pointwiseLinear
        else
            return pointwiseLinear




type InvertedBottleneckBlockStack: Layer {
    let blocks: [InvertedBottleneckBlock] = []

    public init(
        filters: (Int, Int),
        widthMultiplier: double,
        blockCount: int,
        initialStrides: (Int, Int) = (2, 2)
    ) = 
        self.blocks = [
            InvertedBottleneckBlock(
                filters: (filters.0, filters.1), widthMultiplier: widthMultiplier,
                strides: initialStrides)
        ]
        for _ in 1..<blockCount {
            self.blocks.append(
                InvertedBottleneckBlock(
                    filters: (filters.1, filters.1), widthMultiplier: widthMultiplier)
            )



    
    override _.forward(input) =
        return blocks.differentiableReduce(input) =  $1($0)



type MobileNetV2: Layer {
    let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    let inputConv: Conv2D<Float>
    let inputConvBatchNorm: BatchNorm<Float>
    let initialInvertedBottleneck: InitialInvertedBottleneckBlock

    let residualBlockStack1: InvertedBottleneckBlockStack
    let residualBlockStack2: InvertedBottleneckBlockStack
    let residualBlockStack3: InvertedBottleneckBlockStack
    let residualBlockStack4: InvertedBottleneckBlockStack
    let residualBlockStack5: InvertedBottleneckBlockStack

    let invertedBottleneckBlock16: InvertedBottleneckBlock

    let outputConv: Conv2D<Float>
    let outputConvBatchNorm: BatchNorm<Float>
    let avgPool = GlobalAvgPool2D<Float>()
    let outputClassifier: Dense

    public init(classCount: int = 1000, widthMultiplier: double = 1.0) = 
        inputConv = Conv2d(
            filterShape=(3, 3, 3, makeDivisible(filter: 32, widthMultiplier: widthMultiplier)),
            stride=2,
            padding="valid")
        inputConvBatchNorm = BatchNorm(
            featureCount: makeDivisible(filter: 32, widthMultiplier: widthMultiplier))

        initialInvertedBottleneck = InitialInvertedBottleneckBlock(
            filters: (32, 16), widthMultiplier: widthMultiplier)

        residualBlockStack1 = InvertedBottleneckBlockStack(
            filters: (16, 24), widthMultiplier: widthMultiplier, blockCount: 2)
        residualBlockStack2 = InvertedBottleneckBlockStack(
            filters: (24, 32), widthMultiplier: widthMultiplier, blockCount: 3)
        residualBlockStack3 = InvertedBottleneckBlockStack(
            filters: (32, 64), widthMultiplier: widthMultiplier, blockCount: 4)
        residualBlockStack4 = InvertedBottleneckBlockStack(
            filters: (64, 96), widthMultiplier: widthMultiplier, blockCount: 3,
            initialStrides: (1, 1))
        residualBlockStack5 = InvertedBottleneckBlockStack(
            filters: (96, 160), widthMultiplier: widthMultiplier, blockCount: 3)

        invertedBottleneckBlock16 = InvertedBottleneckBlock(
            filters: (160, 320), widthMultiplier: widthMultiplier)

        let lastBlockFilterCount = makeDivisible(filter: 1280, widthMultiplier: widthMultiplier)
        if widthMultiplier < 1 then
            // paper: "One minor implementation difference, with [arxiv:1704.04861] is that for
            // multipliers less than one, we apply width multiplier to all layers except the very
            // last convolutional layer."
            lastBlockFilterCount = 1280


        outputConv = Conv2d(
            filterShape=(
                1, 1,
                makeDivisible(filter: 320, widthMultiplier: widthMultiplier), lastBlockFilterCount
            ),
            stride=1,
            padding="same")
        outputConvBatchNorm = BatchNorm(featureCount=lastBlockFilterCount)

        outputClassifier = Linear(
            inputSize= lastBlockFilterCount, outFeatures=classCount)


    
    override _.forward(input) =
        let convolved = relu6(input |> zeroPad, inputConv, inputConvBatchNorm))
        let initialConv = initialInvertedBottleneck(convolved)
        let backbone = initialConv.sequenced(
            through: residualBlockStack1, residualBlockStack2, residualBlockStack3,
            residualBlockStack4, residualBlockStack5)
        let output = relu6(outputConvBatchNorm(outputConv(invertedBottleneckBlock16(backbone))))
        return output |> avgPool, outputClassifier)


