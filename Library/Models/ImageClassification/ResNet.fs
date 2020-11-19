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

module Models.ImageClassification.ResNet

open DiffSharp
open DiffSharp.Model

type Depth =
    | ResNet18
    | ResNet34
    | ResNet50
    | ResNet56
    | ResNet101
    | ResNet152
    member self.usesBasicBlocks: bool =
        match self with
        | ResNet18 | ResNet34 | ResNet56 -> true
        | _ -> false

    member self.layerBlockSizes =
        match self with
        | ResNet18 -> [| 2; 2; 2; 2 |]
        | ResNet34 -> [| 3; 4; 6; 3 |]
        | ResNet50 -> [| 3; 4; 6; 3 |]
        | ResNet56 -> [| 9; 9; 9 |]
        | ResNet101 -> [| 3; 4; 23; 3 |]
        | ResNet152 -> [| 3; 8; 36; 3 |]


// Original Paper:
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// This uses shortcut layers to connect residual blocks
// (aka Option (B) in https://arxiv.org/abs/1812.01187).
//
// The structure of this implementation was inspired by the Flax ResNet example:
// https://github.com/google/flax/blob/master/examples/imagenet/models.py

type ConvBN(inChannels, outChannels, kernelSize:int, ?stride: int, ?padding: int) =
    inherit Model()

    let conv = Conv2d(inChannels, outChannels, kernelSize=kernelSize, ?stride=stride, ?padding=padding, bias=false)
    let norm = BatchNorm2d(numFeatures=inChannels, momentum=dsharp.scalar 0.9, eps=1e-5)
    
    override _.forward(input) =
        input |> conv.forward |> norm.forward

type ResidualBlock(inputFilters: int, filters: int, stride: int, useLaterStride: bool, isBasic: bool) =
    inherit Model()
    let outFilters = filters * (if isBasic then 1 else 4)
    let needsProjection = (inputFilters <> outFilters) || (stride <> 1)
    let projection =
        // TODO: Replace the following, so as to not waste memory for non-projection cases.
        if needsProjection then
            ConvBN(inputFilters, outFilters, kernelSize=1, stride=stride)
        else
            ConvBN(1, 1, kernelSize=1)

    let earlyConvs =
        if isBasic then
            [| ConvBN(inputFilters, filters, kernelSize=3, stride=stride, padding=1 (* "same " *)) |]
        else
            if useLaterStride then
                // Configure for ResNet V1.5 (the more common implementation).
                [| ConvBN(inputFilters, filters, kernelSize=1)
                   ConvBN(filters, filters, kernelSize=3, stride=stride, padding=1 (* "same " *)) |]
            else
                // Configure for ResNet V1 (the paper implementation).
                [| ConvBN(inputFilters, filters, kernelSize=1, stride=stride)
                   ConvBN(filters, filters, kernelSize=3, padding=1 (* "same " *)) |]

    let lastConv =
        if isBasic then
            ConvBN(filters, outFilters, kernelSize=3, padding=1 (* "same " *))
        else
            ConvBN(filters, outFilters, kernelSize=1)

    override _.forward(input) =
        // TODO: Find a way for this to be checked only at initialization, not during training or 
        // inference.
        let residual =
            if needsProjection then
                projection.forward(input)
            else
                input

        let earlyConvsReduced = 
            (input, earlyConvs) ||> Array.fold (fun last layer -> layer.forward last |> dsharp.relu) 

        let lastConvResult = lastConv.forward(earlyConvsReduced)

        dsharp.relu(lastConvResult + residual)

/// An implementation of the ResNet v1 and v1.5 architectures, at various depths.

/// Initializes a new ResNet v1 or v1.5 network model.
///
/// - Parameters:
///   - classCount: The number of classes the network will be or has been trained to identify.
///   - depth: A specific depth for the network, chosen from the enumerated values in 
///     ResNet.Depth.
///   - downsamplingInFirstStage: Whether or not to downsample by a total of 4X among the first
///     two layers. For ImageNet-sized images, this should be true, but for smaller images like
///     CIFAR-10, this probably should be false for best results.
///   - inputFilters: The number of filters at the first convolution.
///   - useLaterStride: If false, the stride within the residual block is placed at the position
///     specified in He, et al., corresponding to ResNet v1. If true, the stride is moved to the
///     3x3 convolution, corresponding to the v1.5 variant of the architecture. 
type ResNet(classCount: int, depth: Depth, ?downsamplingInFirstStage: bool, ?useLaterStride: bool) =
    inherit Model()
    let downsamplingInFirstStage = defaultArg downsamplingInFirstStage true
    let useLaterStride = defaultArg useLaterStride true
    let avgPool = GlobalAvgPool2d()
    let flatten = Flatten()

    let inputFilters =
        if downsamplingInFirstStage then
            64
        else
            16

    let initialLayer, maxPool =
        if downsamplingInFirstStage then
            ConvBN(3, inputFilters, kernelSize=7, stride=2, padding=3 (* "same " *)),
            MaxPool2d(kernelSize=3, stride=2, padding=1 (* "same " *))
        else
            ConvBN(3, inputFilters, kernelSize=3, padding=1 (* "same " *)),
            MaxPool2d(kernelSize=1, stride=1)  // no-op

    let mutable lastInputChannels = inputFilters
    let residualBlocks =
        [| 
            for (blockSizeIndex, blockSize) in Array.indexed depth.layerBlockSizes do
                for blockIndex in 0..blockSize-1 do
                    let stride = if ((blockSizeIndex > 0) && (blockIndex = 0)) then 2 else 1
                    let filters = inputFilters * int(2.0 ** double blockSizeIndex)
                    let residualBlock = 
                        ResidualBlock(
                            inputFilters=lastInputChannels, filters=filters, stride=stride,
                            useLaterStride=useLaterStride, isBasic=depth.usesBasicBlocks)
                    lastInputChannels <- filters * (if depth.usesBasicBlocks then 1 else 4)
                    residualBlock 
        |]

    let finalFilters = inputFilters * int(2.0 ** double(depth.layerBlockSizes.Length - 1))
    let classifier = 
        Linear(
            inFeatures= (if depth.usesBasicBlocks then finalFilters else finalFilters * 4),
            outFeatures=classCount)
    
    override _.forward(input) =
        let inputLayer = maxPool.forward(dsharp.relu(initialLayer.forward(input)))
        let blocksReduced = (input, residualBlocks) ||> Array.fold (fun last layer -> layer.forward last) 
        blocksReduced |> avgPool.forward |> flatten.forward |> classifier.forward

