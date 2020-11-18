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

namespace Models

open DiffSharp

// Original Paper:
// "Deep Residual Learning for Image Recognition"
// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
// https://arxiv.org/abs/1512.03385
// This uses shortcut layers to connect residual blocks
// (aka Option (B) in https://arxiv.org/abs/1812.01187).
//
// The structure of this implementation was inspired by the Flax ResNet example:
// https://github.com/google/flax/blob/master/examples/imagenet/models.py

type ConvBN() =
    inherit Model()
    let conv: Conv2D<Float>
    let norm: BatchNorm<Float>

    public init(
        kernelSize=(Int, Int, Int, Int),
        strides = [Int, Int) = (1, 1),
        padding: Padding = .valid
    ) = 
        self.conv = Conv2d(filterShape: filterShape, strides=strides, padding: padding, useBias: false)
        self.norm = BatchNorm2d(numFeatures=filterShape.3, momentum: 0.9, epsilon: 1e-5)


    
    override _.forward(input) =
        input |> conv, norm)



type ResidualBlock() =
    inherit Model()
    let projection: ConvBN
    let needsProjection: bool
    let earlyConvs: ConvBN[] = [| |]
    let lastConv: ConvBN

    public init(
        inputFilters: int, filters: int, strides = [Int, Int), useLaterStride: bool, isBasic: bool
    ) = 
        let outFilters = filters * (isBasic ? 1 : 4)
        self.needsProjection = (inputFilters <> outFilters) || (strides.0 <> 1)
        // TODO: Replace the following, so as to not waste memory for non-projection cases.
        if needsProjection then
            projection = ConvBN(kernelSize=(1, 1, inputFilters, outFilters), strides=strides)
        else
            projection = ConvBN(kernelSize=(1, 1, 1, 1))


        if isBasic then
            earlyConvs = [
                (ConvBN(
                    kernelSize=(3, 3, inputFilters, filters), strides=strides, padding=kernelSize/2 (* "same " *))),
            ]
            lastConv = ConvBN(kernelSize=(3, 3, filters, outFilters), padding=kernelSize/2 (* "same " *))
        else
            if useLaterStride then
                // Configure for ResNet V1.5 (the more common implementation).
                earlyConvs.append(ConvBN(kernelSize=(1, 1, inputFilters, filters)))
                earlyConvs.append(
                    ConvBN(kernelSize=(3, 3, filters, filters), strides=strides, padding=kernelSize/2 (* "same " *)))
            else
                // Configure for ResNet V1 (the paper implementation).
                earlyConvs.append(
                    ConvBN(kernelSize=(1, 1, inputFilters, filters), strides=strides))
                earlyConvs.append(ConvBN(kernelSize=(3, 3, filters, filters), padding=kernelSize/2 (* "same " *)))

            lastConv = ConvBN(kernelSize=(1, 1, filters, outFilters))



    
    override _.forward(input) =
        let residual: Tensor
        // TODO: Find a way for this to be checked only at initialization, not during training or 
        // inference.
        if needsProjection then
            residual = projection(input)
        else
            residual = input


        let earlyConvsReduced = earlyConvs.differentiableReduce(input) =  last, layer in
            relu(layer(last))

        let lastConvResult = lastConv(earlyConvsReduced)

        relu(lastConvResult + residual)



/// An implementation of the ResNet v1 and v1.5 architectures, at various depths.
type ResNet() =
    inherit Model()
    let initialLayer: ConvBN
    let maxPool: MaxPool2d
    let residualBlocks: ResidualBlock[] = [| |]
    let avgPool = GlobalAvgPool2d()
    let flatten = Flatten()
    let classifier: Dense

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
    public init(
        classCount: int, depth: Depth, downsamplingInFirstStage: bool = true,
        useLaterStride: bool = true
    ) = 
        let inputFilters: int
        
        if downsamplingInFirstStage then
            inputFilters = 64
            initialLayer = ConvBN(
                kernelSize=(7, 7, 3, inputFilters), stride=2, padding=kernelSize/2 (* "same " *))
            maxPool = MaxPool2D(poolSize: (3, 3), stride=2, padding=kernelSize/2 (* "same " *))
        else
            inputFilters = 16
            initialLayer = ConvBN(kernelSize=(3, 3, 3, inputFilters), padding=kernelSize/2 (* "same " *))
            maxPool = MaxPool2D(poolSize: (1, 1), stride=1)  // no-op


        let lastInputFilterCount = inputFilters
        for (blockSizeIndex, blockSize) in depth.layerBlockSizes.enumerated() do
            for blockIndex in 0..blockSize-1 do
                let strides = ((blockSizeIndex > 0) && (blockIndex = 0)) ? (2, 2) : (1, 1)
                let filters = inputFilters * int(pow(2.0, Double(blockSizeIndex)))
                let residualBlock = ResidualBlock(
                    inputFilters: lastInputFilterCount, filters: filters, strides=strides,
                    useLaterStride: useLaterStride, isBasic: depth.usesBasicBlocks)
                lastInputFilterCount = filters * (depth.usesBasicBlocks ? 1 : 4)
                residualBlocks.append(residualBlock)



        let finalFilters = inputFilters * int(pow(2.0, Double(depth.layerBlockSizes.count - 1)))
        classifier = Linear(
            inputSize= depth.usesBasicBlocks ? finalFilters : finalFilters * 4,
            outputSize=classCount)


    
    override _.forward(input) =
        let inputLayer = maxPool(relu(initialLayer(input)))
        let blocksReduced = residualBlocks.differentiableReduce(inputLayer) =  last, layer in
            layer(last)

        blocksReduced |> avgPool, flatten, classifier)



extension ResNet {
    type Depth {
        | resNet18
        | resNet34
        | resNet50
        | resNet56
        | resNet101
        | resNet152

        let usesBasicBlocks: bool {
            match self with
            | .resNet18, .resNet34, .resNet56 -> true
            | _ -> return false



        let layerBlockSizes: int[] {
            match self with
            | .resNet18 -> return [2, 2, 2, 2]
            | .resNet34 -> return [3, 4, 6, 3]
            | .resNet50 -> return [3, 4, 6, 3]
            | .resNet56 -> return [9, 9, 9]
            | .resNet101 -> return [3, 4, 23, 3]
            | .resNet152 -> return [3, 8, 36, 3]




