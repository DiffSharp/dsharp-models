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

module Models.ImageClassification.WideResNet

open DiffSharp
open DiffSharp.Model

// Original Paper:
// "Wide Residual Networks"
// Sergey Zagoruyko, Nikos Komodakis
// https://arxiv.org/abs/1605.07146
// https://github.com/szagoruyko/wide-residual-networks

type Kind =
    | WideResNet16
    | WideResNet16k8
    | WideResNet16k10
    | WideResNet22
    | WideResNet22k8
    | WideResNet22k10
    | WideResNet28
    | WideResNet28k10
    | WideResNet28k12
    | WideResNet40k1
    | WideResNet40k2
    | WideResNet40k4
    | WideResNet40k8

type BatchNormConv2DBlock(featureCounts: (int * int),
        ?kernelSize: int,
        ?stride: int,
        ?padding: int) =
    inherit Model()
    let kernelSize = defaultArg kernelSize 3
    let padding = defaultArg padding (kernelSize/2)
    let stride = defaultArg stride 1
    let dropout = Dropout2d(p=0.3)
    let featureCounts0, featureCounts1 = featureCounts

    let norm1 = BatchNorm2d(numFeatures=featureCounts0)
    let conv1 = Conv2d(featureCounts0, featureCounts1, kernelSize=kernelSize,  stride=stride, padding=padding)
    let norm2 = BatchNorm2d(numFeatures=featureCounts1)
    let conv2 = Conv2d(featureCounts1, featureCounts1, kernelSize=kernelSize, stride=1, padding=padding)
    let shortcut = Conv2d(featureCounts0, featureCounts1, kernelSize=1, stride=stride, padding=padding)
    let isExpansion = featureCounts1 <> featureCounts0 || stride <> 1
    
    override _.forward(input) =
        let preact1 = dsharp.relu(norm1.forward(input))
        let residual = conv1.forward(preact1)
        let shortcutResult, preact2 =
            if isExpansion then
                shortcut.forward(preact1), dsharp.relu(norm2.forward(residual))
            else 
                input, dropout.forward(dsharp.relu(norm2.forward(residual)))

        let residual = conv2.forward(preact2)
        residual + shortcutResult

type WideResNetBasicBlock(featureCounts: (int * int),
        ?kernelSize: int,
        ?depthFactor: int,
        ?initialStride: int) =
    inherit Model()
    let kernelSize = defaultArg kernelSize 3
    let depthFactor = defaultArg depthFactor 2
    let initialStride = defaultArg initialStride 2
    let _, featureCounts1 = featureCounts
    let blocks =
        [| BatchNormConv2DBlock(featureCounts=featureCounts, stride=initialStride)
           for _ in 1..depthFactor-1 do
              BatchNormConv2DBlock(featureCounts=(featureCounts1, featureCounts1)) |]
  
    override _.forward(input) =
        (input, blocks) ||> Array.fold (fun last layer -> layer.forward last) 

type WideResNet(?depthFactor: int, ?widenFactor: int) =
    inherit Model()
    let depthFactor = defaultArg depthFactor 2
    let widenFactor = defaultArg widenFactor 8
    let flatten = Flatten()

    let l1 = Conv2d(3, 16, kernelSize=3, stride=1, padding=1 (* "same " *))

    let l2 = WideResNetBasicBlock(featureCounts=(16, 16 * widenFactor), depthFactor=depthFactor, initialStride=1)
    let l3 = WideResNetBasicBlock(featureCounts=(16 * widenFactor, 32 * widenFactor), 
                                    depthFactor=depthFactor)
    let l4 = WideResNetBasicBlock(featureCounts=(32 * widenFactor, 64 * widenFactor), 
                                    depthFactor=depthFactor)

    let norm = BatchNorm2d(numFeatures=64 * widenFactor)
    let avgPool = AvgPool2d(kernelSize=8, stride = 8)
    let classifier = Linear(inFeatures=64 * widenFactor, outFeatures=10)
    
    override _.forward(input) =
        let inputLayer = input |> l1.forward |> l2.forward |>  l3.forward |>  l4.forward
        let finalNorm = dsharp.relu(norm.forward(inputLayer))
        finalNorm |> avgPool.forward |> flatten.forward |> classifier.forward

    static member Create (kind: Kind) = 
        match kind with
        | WideResNet16 | WideResNet16k8 ->
            WideResNet(depthFactor=2, widenFactor=8)
        | WideResNet16k10 ->
            WideResNet(depthFactor=2, widenFactor=10)
        | WideResNet22 | WideResNet22k8 ->
            WideResNet(depthFactor=3, widenFactor=8)
        | WideResNet22k10 ->
            WideResNet(depthFactor=3, widenFactor=10)
        | WideResNet28 | WideResNet28k10 ->
            WideResNet(depthFactor=4, widenFactor=10)
        | WideResNet28k12 ->
            WideResNet(depthFactor=4, widenFactor=12)
        | WideResNet40k1 ->
            WideResNet(depthFactor=6, widenFactor=1)
        | WideResNet40k2 ->
            WideResNet(depthFactor=6, widenFactor=2)
        | WideResNet40k4 ->
            WideResNet(depthFactor=6, widenFactor=4)
        | WideResNet40k8 ->
            WideResNet(depthFactor=6, widenFactor=8)
