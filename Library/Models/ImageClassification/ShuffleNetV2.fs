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

// Original V2 paper
// "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
// Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

type ChannelShuffle: ParameterlessLayer {
    type TangentVector = EmptyTangentVector

    let groups: int
    
    public init(groups: int = 2) = 
        self.groups = groups

    
    
    override _.forward(input) =
        let batchSize = input.shape.[0], height = input.shape.[1], width = input.shape.[2],
        channels = input.shape.[3]
        let channelsPerGroup: int = channels / groups
        
        let output = input.view([batchSize, height, width, groups, channelsPerGroup])
        output = output.permute([0, 1, 2, 4, 3])
        output = output.view([batchSize, height, width, channels])
        return output



type InvertedResidual() =
    inherit Model()
    let includeBranch: bool = true
    let zeropad: ZeroPadding2D = ZeroPadding2D<Float>(padding: ((1, 1), (1, 1)))
    
    let branch: Sequential<ZeroPadding2D<Float>, Sequential<DepthwiseConv2D<Float>,
    Sequential<BatchNorm<Float>, Sequential<Conv2D<Float>, BatchNorm<Float>>>>>
    let conv1: Conv2D<Float>
    let batchNorm1: BatchNorm<Float>
    let depthwiseConv: DepthwiseConv2D<Float>
    let batchNorm2: BatchNorm<Float>
    let conv2: Conv2D<Float>
    let batchNorm3: BatchNorm<Float>
    
    public init(filters: (int * int), stride: int) = 
        if stride=1 then
            includeBranch = false

        
        let branchChannels = filters.1 / 2
        branch = Sequential {
            ZeroPadding2D<Float>(padding: ((1, 1), (1, 1)))
            DepthwiseConv2D<Float>(
                filterShape=(3, 3, filters.0, 1), strides = [stride, stride),
                padding="valid"
            )
            BatchNorm(featureCount=filters.0)
            Conv2d(
                filterShape=(1, 1, filters.0, branchChannels), stride=1 (* , padding="valid" *),
                useBias: false
            )
            BatchNorm(featureCount=branchChannels)

        let inputChannels = includeBranch ? filters.0: branchChannels
        conv1 = Conv2d(
            filterShape=(1, 1, inputChannels, branchChannels), stride=1 (* , padding="valid" *),
            useBias: false
        )
        conv2 = Conv2d(
            filterShape=(1, 1, branchChannels, branchChannels), stride=1 (* , padding="valid" *),
            useBias: false
        )
        depthwiseConv = DepthwiseConv2D<Float>(
            filterShape=(3, 3, branchChannels, 1), strides = [stride, stride) (* , padding="valid" *)
        )
        batchNorm1 = BatchNorm(featureCount=branchChannels)
        batchNorm2 = BatchNorm(featureCount=branchChannels)
        batchNorm3 = BatchNorm(featureCount=branchChannels)

    
    
    override _.forward(input) =
        if not includeBranch then
            let splitInput = input.split(count: 2, alongAxis: 3)
            let input1 = splitInput[0]
            let input2 = splitInput[1]
            let output2 = relu(input2 |> conv1, batchNorm1))
            output2 = relu(output2 |> zeropad, depthwiseConv, batchNorm2, conv2,
                                             batchNorm3))
            return ChannelShuffle()(input1.cat(output2, alongAxis: 3))
        else
            let output1 = branch(input)
            let output2 = relu(input |> conv1, batchNorm1))
            output2 = relu(output2 |> zeropad, depthwiseConv, batchNorm2, conv2,
                                             batchNorm3))
            return ChannelShuffle()(output1.cat(output2, alongAxis: 3))






type ShuffleNetV2() =
    inherit Model()
    let zeroPad: ZeroPadding2D<Float> = ZeroPadding2D<Float>(padding: ((1, 1), (1, 1)))
    
    let conv1: Conv2D<Float>
    let batchNorm1: BatchNorm<Float>
    let maxPool: MaxPool2d
    let invertedResidualBlocksStage1: InvertedResidual[]
    let invertedResidualBlocksStage2: InvertedResidual[]
    let invertedResidualBlocksStage3: InvertedResidual[]
    let conv2: Conv2D<Float>
    let globalPool: GlobalAvgPool2D<Float> = GlobalAvgPool2D()
    let dense: Dense
    
    public init(stagesRepeat: (Int, Int, Int), stagesOutputChannels: (Int, Int, Int, Int, Int),
                classCount: int) = 
        let inputChannels = 3
        let outputChannels = stagesOutputChannels.0
        conv1 = Conv2d(
            filterShape=(3, 3, inputChannels, outputChannels), stride=1
        )
        maxPool = MaxPool2D(poolSize: (3, 3), stride=2)
        conv2 = Conv2d(
            filterShape=(1, 1, stagesOutputChannels.3, stagesOutputChannels.4), stride=1,
            useBias: false
        )
        dense = Linear(inFeatures=stagesOutputChannels.4, outFeatures=classCount)
        batchNorm1 = BatchNorm(featureCount=outputChannels)
        inputChannels = outputChannels
        outputChannels = stagesOutputChannels.1
        invertedResidualBlocksStage1 = [InvertedResidual(filters: (inputChannels, outputChannels),
                                                         stride: 2)]
        for _ in 1..stagesRepeat.0 do
            invertedResidualBlocksStage1.append(InvertedResidual(
                filters: (outputChannels, outputChannels), stride=1)
            )

        inputChannels = outputChannels
        outputChannels = stagesOutputChannels.2
        invertedResidualBlocksStage2 = [InvertedResidual(filters: (inputChannels, outputChannels),
                                                         stride: 2)]
        for _ in 1..stagesRepeat.1 do
            invertedResidualBlocksStage2.append(InvertedResidual(
                filters: (outputChannels, outputChannels), stride=1)
            )

        
        inputChannels = outputChannels
        outputChannels = stagesOutputChannels.3
        invertedResidualBlocksStage3 = [InvertedResidual(filters: (inputChannels, outputChannels),
                                                         stride: 2)]
        for _ in 1..stagesRepeat.2 do
            invertedResidualBlocksStage3.append(InvertedResidual(
                filters: (outputChannels, outputChannels), stride=1)
            )


    
    
    override _.forward(input) =
        let output = relu(input |> zeroPad, conv1, batchNorm1, zeroPad, maxPool))
        output = invertedResidualBlocksStage1.differentiableReduce(output) = $1($0)
        output = invertedResidualBlocksStage2.differentiableReduce(output) = $1($0)
        output = invertedResidualBlocksStage3.differentiableReduce(output) = $1($0)
        output = relu(conv2(output))
        return output |> globalPool, dense)



extension ShuffleNetV2 {
    type Kind {
        case shuffleNetV2x05
        case shuffleNetV2x10
        case shuffleNetV2x15
        case shuffleNetV2x20


    public init(kind: Kind) = 
        match kind with
        | .shuffleNetV2x05 ->
            self.init(
                stagesRepeat: (4, 8, 4), stagesOutputChannels: (24, 48, 96, 192, 1024),
                classCount: 1000
            )
        | .shuffleNetV2x10 ->
            self.init(
                stagesRepeat: (4, 8, 4), stagesOutputChannels: (24, 116, 232, 464, 1024),
                classCount: 1000
            )
        | .shuffleNetV2x15 ->
            self.init(
                stagesRepeat: (4, 8, 4), stagesOutputChannels: (24, 176, 352, 704, 1024),
                classCount: 1000
            )
        | .shuffleNetV2x20 ->
            self.init(
                stagesRepeat: (4, 8, 4), stagesOutputChannels: (24, 244, 488, 976, 2048),
                classCount: 1000
            )



