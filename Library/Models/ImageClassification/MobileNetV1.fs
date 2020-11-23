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

module Models.ImageClassification.MobileNetV1

open DiffSharp
open DiffSharp.Model
open System.Diagnostics

// Original Paper:
// "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
// Applications"
// Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko
// Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
// https://arxiv.org/abs/1704.04861

type ConvBlock(filterCount: int, ?widthMultiplier: double, ?stride) =
    inherit Model()
    let widthMultiplier = defaultArg widthMultiplier 1.0
    let zeroPad = ZeroPadding2d(0,1)
    do Debug.Assert(widthMultiplier > 0.0, "Width multiplier must be positive")

    let scaledChannels = int (double filterCount * widthMultiplier)
    let conv = Conv2d(3, scaledChannels, kernelSize=3, ?stride=stride (* , padding="valid" *))
    let batchNorm = BatchNorm2d(numFeatures=scaledChannels)
    
    override _.forward(input) =
        let convolved = input |> zeroPad.forward |> conv.forward |> batchNorm.forward
        dsharp.relu6(convolved)

type DepthwiseConvBlock(filterCount: int, pointwiseChannels: int, stride: int, ?widthMultiplier: double, ?depthMultiplier: int) =
    inherit Model()
    let zeroPad = ZeroPadding2d(0,1)
    let widthMultiplier = defaultArg widthMultiplier 1.0
    let depthMultiplier = defaultArg depthMultiplier 1
    do
        Debug.Assert(widthMultiplier > 0.0, "Width multiplier must be positive")
        Debug.Assert(depthMultiplier > 0, "Depth multiplier must be positive")

    let scaledChannels = int(double(filterCount) * widthMultiplier)
    let scaledPointwiseChannels = int(double(pointwiseChannels) * widthMultiplier)

    let dConv = DepthwiseConv2d(scaledChannels, depthMultiplier, kernelSize=3, stride=stride (* , padding: strides = (1, 1) ? .same : .valid *) )
    let batchNorm1 = BatchNorm2d( scaledChannels * depthMultiplier)
    let conv = Conv2d(scaledChannels * depthMultiplier, scaledPointwiseChannels, kernelSize=1, stride=1, padding=0 (* "same " *))
    let batchNorm2 = BatchNorm2d(numFeatures=scaledPointwiseChannels)
    
    override _.forward(input) =
        let convolved1 =
            if stride = 1 then
                input |> dConv.forward |> batchNorm1.forward
            else
                input |> zeroPad.forward |> dConv.forward |> batchNorm1.forward

        let convolved2 = dsharp.relu6(convolved1)
        let convolved3 = dsharp.relu6(convolved2 |> conv.forward |> batchNorm2.forward)
        convolved3

type MobileNetV1(classCount: int, ?widthMultiplier: double, ?depthMultiplier: int, ?dropout: double) =
    inherit Model()
    let widthMultiplier = defaultArg widthMultiplier 1.0
    let depthMultiplier = defaultArg depthMultiplier 1
    let dropout = defaultArg dropout 0.001
    let avgPool = GlobalAvgPool2d()

    let scaledFilterShape = int(1024.0 * widthMultiplier)

    let convBlock1 = ConvBlock(filterCount=32, widthMultiplier=widthMultiplier, stride=2)
    let dConvBlock1 =
        DepthwiseConvBlock(
            filterCount=32,
            pointwiseChannels=64,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)
    let dConvBlock2 =
        DepthwiseConvBlock(
            filterCount=64,
            pointwiseChannels=128,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=2)
    let dConvBlock3 =
        DepthwiseConvBlock(
            filterCount=128,
            pointwiseChannels=128,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)
    let dConvBlock4 =
        DepthwiseConvBlock(
            filterCount=128,
            pointwiseChannels=256,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=2)
    let dConvBlock5 =
        DepthwiseConvBlock(
            filterCount=256,
            pointwiseChannels=256,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)
    let dConvBlock6 =
        DepthwiseConvBlock(
            filterCount=256,
            pointwiseChannels=512,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=2)
    let dConvBlock7 =
        DepthwiseConvBlock(
            filterCount=512,
            pointwiseChannels=512,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)
    let dConvBlock8 =
        DepthwiseConvBlock(
            filterCount=512,
            pointwiseChannels=512,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)
    let dConvBlock9 =
        DepthwiseConvBlock(
            filterCount=512,
            pointwiseChannels=512,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)
    let dConvBlock10 =
        DepthwiseConvBlock(
            filterCount=512,
            pointwiseChannels=512,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)
    let dConvBlock11 =
        DepthwiseConvBlock(
            filterCount=512,
            pointwiseChannels=512,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)
    let dConvBlock12 = 
        DepthwiseConvBlock(
            filterCount=512,
            pointwiseChannels=1024,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=2)
    let dConvBlock13 = 
        DepthwiseConvBlock(
            filterCount=1024,
            pointwiseChannels=1024,
            widthMultiplier=widthMultiplier,
            depthMultiplier=depthMultiplier,
            stride=1)

    let dropoutLayer = Dropout(dropout)
    let convLast = Conv2d(scaledFilterShape, classCount, kernelSize=1, stride=1, padding=0 (* "same " *))

    override _.forward(input) =
        let convolved = input.sequenced(convBlock1, dConvBlock1, dConvBlock2, dConvBlock3, dConvBlock4)
        let convolved2 = convolved.sequenced(dConvBlock5, dConvBlock6, dConvBlock7, dConvBlock8, dConvBlock9)
        let convolved3 = convolved2.sequenced(dConvBlock10, dConvBlock11, dConvBlock12, dConvBlock13, avgPool).view([
                input.shape.[0]; 1; 1; scaledFilterShape
            ])
        let convolved4 = convolved3 |> dropoutLayer.forward |> convLast.forward
        let output = convolved4.view([input.shape.[0]; classCount])
        output


