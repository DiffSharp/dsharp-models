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
// "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
// Applications"
// Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko
// Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
// https://arxiv.org/abs/1704.04861

type ConvBlock() =
    inherit Model()
    let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    let conv: Conv2D<Float>
    let batchNorm: BatchNorm<Float>

    public init(filterCount: int, widthMultiplier: double = 1.0, strides = [Int, Int)) = 
        Debug.Assert(widthMultiplier > 0, "Width multiplier must be positive")

        let scaledFilterCount: int = int(double(filterCount) * widthMultiplier)

        conv = Conv2d(
            filterShape=(3, 3, 3, scaledFilterCount),
            strides: strides,
            padding="valid")
        batchNorm = BatchNorm(featureCount=scaledFilterCount)


    
    override _.forward(input) =
        let convolved = input |> zeroPad, conv, batchNorm)
        return relu6(convolved)



type DepthwiseConvBlock() =
    inherit Model()
    @noDerivative
    let strides = [Int, Int)

    @noDerivative
    let zeroPad = ZeroPadding2D<Float>(padding: ((0, 1), (0, 1)))
    let dConv: DepthwiseConv2D<Float>
    let batchNorm1: BatchNorm<Float>
    let conv: Conv2D<Float>
    let batchNorm2: BatchNorm<Float>

    public init(
        filterCount: int, pointwiseFilterCount: int, widthMultiplier: double = 1.0,
        depthMultiplier: int, strides = [Int, Int)
    ) = 
        Debug.Assert(widthMultiplier > 0, "Width multiplier must be positive")
        Debug.Assert(depthMultiplier > 0, "Depth multiplier must be positive")

        self.strides = strides

        let scaledFilterCount = int(double(filterCount) * widthMultiplier)
        let scaledPointwiseFilterCount = int(double(pointwiseFilterCount) * widthMultiplier)

        dConv = DepthwiseConv2D<Float>(
            filterShape=(3, 3, scaledFilterCount, depthMultiplier),
            strides: strides,
            padding: strides = (1, 1) ? .same : .valid)
        batchNorm1 = BatchNorm(
            featureCount: scaledFilterCount * depthMultiplier)
        conv = Conv2d(
            filterShape=(
                1, 1, scaledFilterCount * depthMultiplier,
                scaledPointwiseFilterCount
            ),
            stride=1,
            padding="same")
        batchNorm2 = BatchNorm(featureCount=scaledPointwiseFilterCount)


    
    override _.forward(input) =
        let convolved1: Tensor
        if self.strides = (1, 1) = 
            convolved1 = input |> dConv, batchNorm1)
        else
            convolved1 = input |> zeroPad, dConv, batchNorm1)

        let convolved2 = relu6(convolved1)
        let convolved3 = relu6(convolved2 |> conv, batchNorm2))
        return convolved3



type MobileNetV1() =
    inherit Model()
    let classCount: int
    let scaledFilterShape: int

    let convBlock1: ConvBlock
    let dConvBlock1: DepthwiseConvBlock
    let dConvBlock2: DepthwiseConvBlock
    let dConvBlock3: DepthwiseConvBlock
    let dConvBlock4: DepthwiseConvBlock
    let dConvBlock5: DepthwiseConvBlock
    let dConvBlock6: DepthwiseConvBlock
    let dConvBlock7: DepthwiseConvBlock
    let dConvBlock8: DepthwiseConvBlock
    let dConvBlock9: DepthwiseConvBlock
    let dConvBlock10: DepthwiseConvBlock
    let dConvBlock11: DepthwiseConvBlock
    let dConvBlock12: DepthwiseConvBlock
    let dConvBlock13: DepthwiseConvBlock
    let avgPool = GlobalAvgPool2D<Float>()
    let dropoutLayer: Dropout<Float>
    let convLast: Conv2D<Float>

    public init(
        classCount: int, widthMultiplier: double = 1.0, depthMultiplier: int = 1,
        dropout: Double = 0.001
    ) = 
        self.classCount = classCount
        scaledFilterShape = int(1024.0 * widthMultiplier)

        convBlock1 = ConvBlock(filterCount: 32, widthMultiplier: widthMultiplier, stride=2)
        dConvBlock1 = DepthwiseConvBlock(
            filterCount: 32,
            pointwiseFilterCount: 64,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)
        dConvBlock2 = DepthwiseConvBlock(
            filterCount: 64,
            pointwiseFilterCount: 128,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=2)
        dConvBlock3 = DepthwiseConvBlock(
            filterCount: 128,
            pointwiseFilterCount: 128,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)
        dConvBlock4 = DepthwiseConvBlock(
            filterCount: 128,
            pointwiseFilterCount: 256,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=2)
        dConvBlock5 = DepthwiseConvBlock(
            filterCount: 256,
            pointwiseFilterCount: 256,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)
        dConvBlock6 = DepthwiseConvBlock(
            filterCount: 256,
            pointwiseFilterCount: 512,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=2)
        dConvBlock7 = DepthwiseConvBlock(
            filterCount: 512,
            pointwiseFilterCount: 512,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)
        dConvBlock8 = DepthwiseConvBlock(
            filterCount: 512,
            pointwiseFilterCount: 512,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)
        dConvBlock9 = DepthwiseConvBlock(
            filterCount: 512,
            pointwiseFilterCount: 512,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)
        dConvBlock10 = DepthwiseConvBlock(
            filterCount: 512,
            pointwiseFilterCount: 512,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)
        dConvBlock11 = DepthwiseConvBlock(
            filterCount: 512,
            pointwiseFilterCount: 512,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)
        dConvBlock12 = DepthwiseConvBlock(
            filterCount: 512,
            pointwiseFilterCount: 1024,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=2)
        dConvBlock13 = DepthwiseConvBlock(
            filterCount: 1024,
            pointwiseFilterCount: 1024,
            widthMultiplier: widthMultiplier,
            depthMultiplier: depthMultiplier,
            stride=1)

        dropoutLayer = Dropout(probability: dropout)
        convLast = Conv2d(
            filterShape=(1, 1, scaledFilterShape, classCount),
            stride=1,
            padding="same")


    
    override _.forward(input) =
        let convolved = input.sequenced(
            through: convBlock1, dConvBlock1,
            dConvBlock2, dConvBlock3, dConvBlock4)
        let convolved2 = convolved.sequenced(
            through: dConvBlock5, dConvBlock6,
            dConvBlock7, dConvBlock8, dConvBlock9)
        let convolved3 = convolved2.sequenced(
            through: dConvBlock10, dConvBlock11, dConvBlock12, dConvBlock13, avgPool).view([
                input.shape.[0], 1, 1, scaledFilterShape
            ])
        let convolved4 = convolved3 |> dropoutLayer, convLast)
        let output = convolved4.view([input.shape.[0], classCount])
        return output


