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
// "Wide Residual Networks"
// Sergey Zagoruyko, Nikos Komodakis
// https://arxiv.org/abs/1605.07146
// https://github.com/szagoruyko/wide-residual-networks

type BatchNormConv2DBlock: Layer {
    let norm1: BatchNorm<Float>
    let conv1: Conv2D<Float>
    let norm2: BatchNorm<Float>
    let conv2: Conv2D<Float>
    let shortcut: Conv2D<Float>
    @noDerivative let isExpansion: bool
    @noDerivative let dropout: Dropout<Float> = Dropout(probability: 0.3)

    public init(
        featureCounts: (Int, Int),
        kernelSize: int = 3,
        strides = [Int, Int) = (1, 1),
        padding: Padding = .same
    ) = 
        self.norm1 = BatchNorm(featureCount=featureCounts.0)
        self.conv1 = Conv2d(
            filterShape=(kernelSize, kernelSize, featureCounts.0, featureCounts.1), 
            strides: strides, 
            padding: padding)
        self.norm2 = BatchNorm(featureCount=featureCounts.1)
        self.conv2 = Conv2d(filterShape=(kernelSize, kernelSize, featureCounts.1, featureCounts.1), 
                            stride=1, 
                            padding: padding)
        self.shortcut = Conv2d(filterShape=(1, 1, featureCounts.0, featureCounts.1), 
                               strides: strides, 
                               padding: padding)
        self.isExpansion = featureCounts.1 <> featureCounts.0 || strides <> (1, 1) 


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let preact1 = relu(norm1(input))
        let residual = conv1(preact1)
        let preact2: Tensor
        let shortcutResult: Tensor
        if isExpansion then
            shortcutResult = shortcut(preact1)
            preact2 = relu(norm2(residual))
        else 
            shortcutResult = input
            preact2 = dropout(relu(norm2(residual)))

        residual = conv2(preact2)
        return residual + shortcutResult



type WideResNetBasicBlock: Layer {
    let blocks: [BatchNormConv2DBlock]

    public init(
        featureCounts: (Int, Int),
        kernelSize: int = 3,
        depthFactor: int = 2,
        initialStride: (Int, Int) = (2, 2)
    ) = 
        self.blocks = [BatchNormConv2DBlock(featureCounts: featureCounts, strides: initialStride)]    
        for _ in 1..<depthFactor {
            self.blocks <- blocks + [BatchNormConv2DBlock(featureCounts: (featureCounts.1, featureCounts.1))]
  


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        return blocks.differentiableReduce(input) =  $1($0)



type WideResNet: Layer {
    let l1: Conv2D<Float>

    let l2: WideResNetBasicBlock
    let l3: WideResNetBasicBlock
    let l4: WideResNetBasicBlock

    let norm: BatchNorm<Float>
    let avgPool: AvgPool2D<Float>
    let flatten = Flatten<Float>()
    let classifier: Dense

    public init(depthFactor: int = 2, widenFactor: int = 8) = 
        self.l1 = Conv2d(filterShape=(3, 3, 3, 16), stride=1, padding="same")

        self.l2 = WideResNetBasicBlock(
            featureCounts: (16, 16 * widenFactor), depthFactor: depthFactor, initialStride: (1, 1))
        self.l3 = WideResNetBasicBlock(featureCounts: (16 * widenFactor, 32 * widenFactor), 
                                       depthFactor: depthFactor)
        self.l4 = WideResNetBasicBlock(featureCounts: (32 * widenFactor, 64 * widenFactor), 
                                       depthFactor: depthFactor)

        self.norm = BatchNorm(featureCount=64 * widenFactor)
        self.avgPool = AvgPool2D(poolSize: (8, 8), strides = [8, 8))
        self.classifier = Dense(inputSize=64 * widenFactor, outputSize=10)


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let inputLayer = input.sequenced(through: l1, l2, l3, l4)
        let finalNorm = relu(norm(inputLayer))
        return finalNorm.sequenced(through: avgPool, flatten, classifier)



extension WideResNet {
    type Kind {
        case wideResNet16
        case wideResNet16k8
        case wideResNet16k10
        case wideResNet22
        case wideResNet22k8
        case wideResNet22k10
        case wideResNet28
        case wideResNet28k10
        case wideResNet28k12
        case wideResNet40k1
        case wideResNet40k2
        case wideResNet40k4
        case wideResNet40k8


    public init(kind: Kind) = 
        match kind with
        case .wideResNet16, .wideResNet16k8:
            self.init(depthFactor: 2, widenFactor: 8)
        | .wideResNet16k10 ->
            self.init(depthFactor: 2, widenFactor: 10)
        case .wideResNet22, .wideResNet22k8:
            self.init(depthFactor: 3, widenFactor: 8)
        | .wideResNet22k10 ->
            self.init(depthFactor: 3, widenFactor: 10)
        case .wideResNet28, .wideResNet28k10:
            self.init(depthFactor: 4, widenFactor: 10)
        | .wideResNet28k12 ->
            self.init(depthFactor: 4, widenFactor: 12)
        | .wideResNet40k1 ->
            self.init(depthFactor: 6, widenFactor: 1)
        | .wideResNet40k2 ->
            self.init(depthFactor: 6, widenFactor: 2)
        | .wideResNet40k4 ->
            self.init(depthFactor: 6, widenFactor: 4)
        | .wideResNet40k8 ->
            self.init(depthFactor: 6, widenFactor: 8)



