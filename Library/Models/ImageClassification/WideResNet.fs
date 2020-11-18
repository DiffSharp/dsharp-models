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
// "Wide Residual Networks"
// Sergey Zagoruyko, Nikos Komodakis
// https://arxiv.org/abs/1605.07146
// https://github.com/szagoruyko/wide-residual-networks

type BatchNormConv2DBlock() =
    inherit Model()
    let norm1: BatchNorm<Float>
    let conv1: Conv2d
    let norm2: BatchNorm<Float>
    let conv2: Conv2d
    let shortcut: Conv2d
    let isExpansion: bool
    let dropout: Dropout<Float> = Dropout2d(p=0.3)

    public init(
        featureCounts: (int * int),
        kernelSize: int = 3,
        strides = [Int, Int) = (1, 1),
        padding: Padding = .same
    ) = 
        self.norm1 = BatchNorm2d(numFeatures=featureCounts.0)
        self.conv1 = Conv2d(
            kernelSize=(kernelSize, kernelSize, featureCounts.0, featureCounts.1), 
            strides=strides, 
            padding: padding)
        self.norm2 = BatchNorm2d(numFeatures=featureCounts.1)
        self.conv2 = Conv2d(kernelSize=(kernelSize, kernelSize, featureCounts.1, featureCounts.1), 
                            stride=1, 
                            padding: padding)
        self.shortcut = Conv2d(kernelSize=(1, 1, featureCounts.0, featureCounts.1), 
                               strides=strides, 
                               padding: padding)
        self.isExpansion = featureCounts.1 <> featureCounts.0 || strides <> (1, 1) 


    
    override _.forward(input) =
        let preact1 = dsharp.relu(norm1(input))
        let residual = conv1.forward(preact1)
        let preact2: Tensor
        let shortcutResult: Tensor
        if isExpansion then
            shortcutResult = shortcut(preact1)
            preact2 = dsharp.relu(norm2(residual))
        else 
            shortcutResult = input
            preact2 = dropout(dsharp.relu(norm2(residual)))

        residual = conv2.forward(preact2)
        residual + shortcutResult



type WideResNetBasicBlock() =
    inherit Model()
    let blocks: BatchNormConv2DBlock[]

    public init(
        featureCounts: (int * int),
        kernelSize: int = 3,
        depthFactor: int = 2,
        initialStride: (int * int) = (2, 2)
    ) = 
        self.blocks = [BatchNormConv2DBlock(featureCounts: featureCounts, strides: initialStride)]    
        for _ in 1..depthFactor-1 do
            self.blocks <- blocks + [BatchNormConv2DBlock(featureCounts: (featureCounts.1, featureCounts.1))]
  


    
    override _.forward(input) =
        blocks.differentiableReduce(input) =  $1($0)



type WideResNet() =
    inherit Model()
    let l1: Conv2d

    let l2: WideResNetBasicBlock
    let l3: WideResNetBasicBlock
    let l4: WideResNetBasicBlock

    let norm: BatchNorm<Float>
    let avgPool: AvgPool2d<Float>
    let flatten = Flatten()
    let classifier: Dense

    public init(depthFactor: int = 2, widenFactor: int = 8) = 
        self.l1 = Conv2d(kernelSize=(3, 3, 3, 16), stride=1, padding=kernelSize/2 (* "same " *))

        self.l2 = WideResNetBasicBlock(
            featureCounts: (16, 16 * widenFactor), depthFactor: depthFactor, initialStride: (1, 1))
        self.l3 = WideResNetBasicBlock(featureCounts: (16 * widenFactor, 32 * widenFactor), 
                                       depthFactor: depthFactor)
        self.l4 = WideResNetBasicBlock(featureCounts: (32 * widenFactor, 64 * widenFactor), 
                                       depthFactor: depthFactor)

        self.norm = BatchNorm2d(numFeatures=64 * widenFactor)
        self.avgPool = AvgPool2d(poolSize: (8, 8), strides = [8, 8))
        self.classifier = Linear(inFeatures=64 * widenFactor, outFeatures=10)


    
    override _.forward(input) =
        let inputLayer = input |> l1, l2, l3, l4)
        let finalNorm = dsharp.relu(norm(inputLayer))
        finalNorm |> avgPool, flatten, classifier)



extension WideResNet {
    type Kind {
        | wideResNet16
        | wideResNet16k8
        | wideResNet16k10
        | wideResNet22
        | wideResNet22k8
        | wideResNet22k10
        | wideResNet28
        | wideResNet28k10
        | wideResNet28k12
        | wideResNet40k1
        | wideResNet40k2
        | wideResNet40k4
        | wideResNet40k8


    public init(kind: Kind) = 
        match kind with
        | .wideResNet16, .wideResNet16k8:
            self.init(depthFactor: 2, widenFactor: 8)
        | .wideResNet16k10 ->
            self.init(depthFactor: 2, widenFactor: 10)
        | .wideResNet22, .wideResNet22k8:
            self.init(depthFactor: 3, widenFactor: 8)
        | .wideResNet22k10 ->
            self.init(depthFactor: 3, widenFactor: 10)
        | .wideResNet28, .wideResNet28k10:
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



