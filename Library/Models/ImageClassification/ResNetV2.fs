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
// V2 paper
// "Bag of Tricks for Image Classification with Convolutional Neural Networks"
// Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
// https://arxiv.org/abs/1812.01187

type Depth =
    | ResNet18
    | ResNet34
    | ResNet50
    | ResNet101
    | ResNet152

    member self.expansion: int =
        match self with
        | ResNet18 | ResNet34 -> 1
        | _ -> 4

    member self.layerBlockSizes: int[] =
        match self with
        | ResNet18 -> [| 2; 2; 2;  2 |]
        | ResNet34 -> [| 3; 4; 6;  3 |]
        | ResNet50 -> [| 3; 4; 6;  3 |]
        | ResNet101 -> [| 3; 4; 23; 3 |]
        | ResNet152 -> [| 3; 8; 36; 3 |]

// A convolution and batchnorm layer
type ConvBNV2(inFilters: int,
        outFilters: int,
        kernelSize: int = 1,
        stride: int = 1,
        padding: Padding = .same,
        isLast: bool = false) =
    inherit Model()

    self.conv = Conv2d(
        kernelSize=(kernelSize, kernelSize, inFilters, outFilters), 
        strides = [stride, stride), 
        padding: padding,
        useBias: false)
    self.isLast = isLast
    if isLast then
        //Initialize the last BatchNorm layer to scale zero
        self.norm = BatchNorm2d((
                axis = -1, 
                momentum: 0.9, 
                offset: dsharp.zeros([outFilters]),
                scale: dsharp.zeros([outFilters]),
                epsilon: 1e-5,
                runningMean: dsharp.tensor(0),
                runningVariance: dsharp.tensor(1))
    else
        self.norm = BatchNorm2d(numFeatures=outFilters, momentum: 0.9, epsilon: 1e-5)


    override _.forward(input) =
        let convResult = input |> conv, norm)
        isLast ? convResult : dsharp.relu(convResult)



// The shortcut in a Residual Block
// Workaround optionals not being differentiable, can be simplified when it's the case
// Resnet-D trick: use average pooling instead of stride 2 conv for the shortcut
type Shortcut(inFilters: int, outFilters: int, stride: int) =
    inherit Model()
    
    let avgPool = AvgPool2d<Float>(kernelSize=2, strides = [stride, stride))
    let needsPool = (stride <> 1)
    let needsProjection = (inFilters <> outFilters)
    let projection = ConvBNV2(
        inFilters: needsProjection ? inFilters  : 1, 
        outFilters: needsProjection ? outFilters : 1
    )

    override _.forward(input) =
        let res = input
        if needsProjection then res = projection(res)
        if needsPool       then res = avgPool(res)
        res

// Residual block for a ResNet V2
// Resnet-B trick: stride on the inside conv
type ResidualBlockV2(inFilters: int, outFilters: int, stride: int, expansion: int) =
    inherit Model()

    if expansion = 1 then
        convs = [
            ConvBNV2(inFilters: inFilters,  outFilters: outFilters, kernelSize: 3, stride: stride),
            ConvBNV2(inFilters: outFilters, outFilters: outFilters, kernelSize: 3, isLast: true)
        ]
    else
        convs = [
            ConvBNV2(inFilters: inFilters,    outFilters: outFilters/4),
            ConvBNV2(inFilters: outFilters/4, outFilters: outFilters/4, kernelSize: 3, stride: stride),
            ConvBNV2(inFilters: outFilters/4, outFilters: outFilters, isLast: true)
        ]

    shortcut = Shortcut(inFilters: inFilters, outFilters: outFilters, stride: stride)

   
    override _.forward(input) =
        let convResult = convs.differentiableReduce(input) =  $1($0)
        dsharp.relu(convResult + shortcut(input))

/// An implementation of the ResNet v2 architectures, at various depths.
///
/// Initializes a new ResNet v2 network model.
///
/// - Parameters:
///   - classCount: The number of classes the network will be or has been trained to identify.
///   - depth: A specific depth for the network, chosen from the enumerated values in 
///     ResNet.Depth.
///   - inputChannels: The number of channels of the input
///   - stemFilters: The number of filters in the first three convolutions.
///         Resnet-A trick uses 64-64-64, research at fastai suggests 32-32-64 is better
type ResNetV2(classCount: int, 
        depth: Depth, 
        inputChannels: int = 3, 
        stemFilters: int[] = [32, 32, 64]) =
    inherit Model()
    let avgPool = GlobalAvgPool2d()
    let flatten = Flatten()

    let filters = [inputChannels] + stemFilters
    inputStem = Array(0..<3).map { i in
        ConvBNV2(inFilters: filters[i], outFilters: filters[i+1], kernelSize: 3, stride: i==0 ? 2 : 1)

    maxPool = MaxPool2d((3, 3), stride=2, padding=kernelSize/2 (* "same " *))
    let sizes = [64 / depth.expansion, 64, 128, 256, 512]
    for (iBlock, nBlocks) in depth.layerBlockSizes.enumerated() do
        let (nIn, nOut) = (sizes[iBlock] * depth.expansion, sizes[iBlock+1] * depth.expansion)
        for j in 0..nBlocks-1 do
            residualBlocks.append(ResidualBlockV2(
                inFilters: j==0 ? nIn : nOut,  
                outFilters: nOut, 
                stride: (iBlock <> 0) && (j = 0) ? 2 : 1, 
                expansion: depth.expansion
            ))


    classifier = Linear(inFeatures=512 * depth.expansion, outFeatures=classCount)

    override _.forward(input) =
        let inputLayer = maxPool(inputStem.differentiableReduce(input) =  $1($0))
        let blocksReduced = residualBlocks.differentiableReduce(inputLayer) =  $1($0)
        blocksReduced |> avgPool, flatten, classifier)
