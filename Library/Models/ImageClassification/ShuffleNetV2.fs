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

module Models.ImageClassification.ShuffleNetV2

open DiffSharp
open DiffSharp.Model

// Original V2 paper
// "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
// Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

type Kind =
    | ShuffleNetV2x05
    | ShuffleNetV2x10
    | ShuffleNetV2x15
    | ShuffleNetV2x20

type ChannelShuffle(?groups: int) =
    inherit Model()
    let groups = defaultArg groups 2
    
    override _.forward(input) =
        let batchSize = input.shape.[0]
        let height = input.shape.[1]
        let width = input.shape.[2]
        let channels = input.shape.[3]
        let channelsPerGroup = channels / groups
        
        let output = input.view([batchSize; height; width; groups; channelsPerGroup])
        let output = output.permute([| 0; 1; 2; 4; 3 |])
        let output = output.view([| batchSize; height; width; channels |])
        output

type InvertedResidual(filters: (int * int), stride: int) =
    inherit Model()
    let includeBranch = (stride<>1)
    let zeropad = ZeroPadding2d(1,1)

    let filters0, filters1 = filters
    let branchChannels = filters1 / 2
    let branch = 
        Sequential [
            ZeroPadding2d(1,1)
            DepthwiseConv2d(filters0, 1, kernelSize=3, stride = stride (* ,padding="valid" *))
            BatchNorm2d(numFeatures=filters0)
            Conv2d(filters0, branchChannels, kernelSize=1, stride=1 (* , padding="valid" *),bias=false)
            BatchNorm2d(numFeatures=branchChannels)
        ]

    let inChannels = if includeBranch then filters0 else branchChannels
    let conv1 = Conv2d(inChannels, branchChannels, kernelSize=1, stride=1 (* , padding="valid" *), bias=false)
    let conv2 = Conv2d(branchChannels, branchChannels, kernelSize=1, stride=1 (* , padding="valid" *), bias=false)
    let depthwiseConv = DepthwiseConv2d(branchChannels, 1, kernelSize=3, stride=stride (* , padding="valid" *))
    let batchNorm1 = BatchNorm2d(numFeatures=branchChannels)
    let batchNorm2 = BatchNorm2d(numFeatures=branchChannels)
    let batchNorm3 = BatchNorm2d(numFeatures=branchChannels)

    override _.forward(input) =
        if not includeBranch then
            let splitInput = input.chunk(count=2, dim=3)
            let input1 = splitInput.[0]
            let input2 = splitInput.[1]
            let output2 = dsharp.relu(input2 |> conv1.forward |> batchNorm1.forward)
            let output2 = dsharp.relu(output2 |> zeropad.forward |> depthwiseConv.forward |> batchNorm2.forward |> conv2.forward |> batchNorm3.forward)
            ChannelShuffle().forward(input1.cat(output2, dim=3))
        else
            let output1 = branch.forward(input)
            let output2 = dsharp.relu(input |> conv1.forward |> batchNorm1.forward)
            let output2 = dsharp.relu(output2 |> zeropad.forward |> depthwiseConv.forward |> batchNorm2.forward |> conv2.forward |> batchNorm3.forward)
            ChannelShuffle().forward(output1.cat(output2, dim=3))

type ShuffleNetV2(stagesRepeat, stagesOutputChannels, classCount: int) =
    inherit Model()
    let zeroPad = ZeroPadding2d(1,1)
    
    let (stagesRepeat0, stagesRepeat1, stagesRepeat2) = stagesRepeat
    let (stagesOutputChannels0, stagesOutputChannels1, stagesOutputChannels2, stagesOutputChannels3, stagesOutputChannels4) = stagesOutputChannels
    let globalPool = GlobalAvgPool2d()
    
    let inChannels = 3
    let outChannels = stagesOutputChannels0
    let conv1 = Conv2d(inChannels, outChannels, kernelSize=3, stride=1)
    let maxPool = MaxPool2d(kernelSize=3, stride=2)
    let conv2 = Conv2d(stagesOutputChannels3, stagesOutputChannels4, kernelSize=1, stride=1,bias=false)
    let dense = Linear(inFeatures=stagesOutputChannels4, outFeatures=classCount)
    let batchNorm1 = BatchNorm2d(numFeatures=outChannels)
    let inChannels = outChannels
    let outChannels = stagesOutputChannels1
    let invertedResidualBlocksStage1 = 
        [| InvertedResidual(filters=(inChannels, outChannels),stride=2)
           for _ in 1..stagesRepeat0 do
                InvertedResidual(filters=(outChannels, outChannels), stride=1) |]

    let inChannels = outChannels
    let outChannels = stagesOutputChannels2
    let invertedResidualBlocksStage2 = 
        [| InvertedResidual(filters=(inChannels, outChannels),stride=2)
           for _ in 1..stagesRepeat1 do
                InvertedResidual(filters=(outChannels, outChannels), stride=1) |]

    let inChannels = outChannels
    let outChannels = stagesOutputChannels3
    let invertedResidualBlocksStage3 = 
        [| InvertedResidual(filters=(inChannels, outChannels), stride=2)
           for _ in 1..stagesRepeat2 do
                InvertedResidual(filters=(outChannels, outChannels), stride=1) |]
    
    override _.forward(input) =
        let output = dsharp.relu(input |> zeroPad.forward |> conv1.forward |> batchNorm1.forward |> zeroPad.forward |> maxPool.forward)
        let output = (output, invertedResidualBlocksStage1) ||> Array.fold (fun last layer -> layer.forward last) 
        let output = (output, invertedResidualBlocksStage2) ||> Array.fold (fun last layer -> layer.forward last) 
        let output = (output, invertedResidualBlocksStage3) ||> Array.fold (fun last layer -> layer.forward last) 
        let output = dsharp.relu(conv2.forward(output))
        output |> globalPool.forward |> dense.forward

    static member Create (kind: Kind) = 
        match kind with
        | ShuffleNetV2x05 ->
            ShuffleNetV2(
                stagesRepeat=(4, 8, 4), stagesOutputChannels=(24, 48, 96, 192, 1024),
                classCount=1000
            )
        | ShuffleNetV2x10 ->
            ShuffleNetV2(
                stagesRepeat=(4, 8, 4), stagesOutputChannels=(24, 116, 232, 464, 1024),
                classCount=1000
            )
        | ShuffleNetV2x15 ->
            ShuffleNetV2(
                stagesRepeat=(4, 8, 4), stagesOutputChannels=(24, 176, 352, 704, 1024),
                classCount=1000
            )
        | ShuffleNetV2x20 ->
            ShuffleNetV2(
                stagesRepeat=(4, 8, 4), stagesOutputChannels=(24, 244, 488, 976, 2048),
                classCount=1000
            )



