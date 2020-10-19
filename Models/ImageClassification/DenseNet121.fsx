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
// Densely Connected Convolutional Networks
// Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
// https://arxiv.org/pdf/1608.06993.pdf

type DenseNet121: Layer {
    let conv = Conv(
        filterSize: 7,
        stride: 2,
        inputFilterCount: 3,
        outputFilterCount: 64
    )
    let maxpool = MaxPool2D<Float>(
        poolSize: (3, 3),
        stride=2,
        padding="same"
    )
    let denseBlock1 = DenseBlock(repetitionCount: 6, inputFilterCount: 64)
    let transitionLayer1 = TransitionLayer(inputFilterCount: 256)
    let denseBlock2 = DenseBlock(repetitionCount: 12, inputFilterCount: 128)
    let transitionLayer2 = TransitionLayer(inputFilterCount: 512)
    let denseBlock3 = DenseBlock(repetitionCount: 24, inputFilterCount: 256)
    let transitionLayer3 = TransitionLayer(inputFilterCount: 1024)
    let denseBlock4 = DenseBlock(repetitionCount: 16, inputFilterCount: 512)
    let globalAvgPool = GlobalAvgPool2D<Float>()
    let dense: Dense

    public init(classCount: int) = 
        dense = Dense(inputSize=1024, outputSize=classCount)


    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        let inputLayer = input.sequenced(through: conv, maxpool)
        let level1 = inputLayer.sequenced(through: denseBlock1, transitionLayer1)
        let level2 = level1.sequenced(through: denseBlock2, transitionLayer2)
        let level3 = level2.sequenced(through: denseBlock3, transitionLayer3)
        let output = level3.sequenced(through: denseBlock4, globalAvgPool, dense)
        return output



extension DenseNet121 {
    type Conv: Layer {
        let batchNorm: BatchNorm<Float>
        let conv: Conv2D<Float>

        public init(
            filterSize: int,
            stride: int = 1,
            inputFilterCount: int,
            outputFilterCount: int
        ) = 
            batchNorm = BatchNorm(featureCount=inputFilterCount)
            conv = Conv2d(
                filterShape=(filterSize, filterSize, inputFilterCount, outputFilterCount),
                strides = [stride, stride),
                padding="same"
            )


        @differentiable
        member _.forward(input: Tensor) : Tensor (* <Float> *) {
            conv(relu(batchNorm(input)))



    /// A pair of a 1x1 `Conv` layer and a 3x3 `Conv` layer.
    type ConvPair: Layer {
        let conv1x1: Conv
        let conv3x3: Conv

        public init(inputFilterCount: int, growthRate: int) = 
            conv1x1 = Conv(
                filterSize: 1,
                inputFilterCount: inputFilterCount,
                outputFilterCount: inputFilterCount * 2
            )
            conv3x3 = Conv(
                filterSize: 3,
                inputFilterCount: inputFilterCount * 2,
                outputFilterCount: growthRate
            )


        @differentiable
        member _.forward(input: Tensor) : Tensor (* <Float> *) {
            let conv1Output = conv1x1(input)
            let conv3Output = conv3x3(conv1Output)
            return conv3Output.concatenated(input, alongAxis: -1)



    type DenseBlock: Layer {
        let pairs: [ConvPair] = []

        public init(repetitionCount: int, growthRate: int = 32, inputFilterCount: int) = 
            for i in 0..<repetitionCount {
                let filterCount = inputFilterCount + i * growthRate
                pairs.append(ConvPair(inputFilterCount: filterCount, growthRate: growthRate))



        @differentiable
        member _.forward(input: Tensor) : Tensor (* <Float> *) {
            pairs.differentiableReduce(input) =  last, layer in
                layer(last)




    type TransitionLayer: Layer {
        let conv: Conv
        let pool: AvgPool2D<Float>

        public init(inputFilterCount: int) = 
            conv = Conv(
                filterSize: 1,
                inputFilterCount: inputFilterCount,
                outputFilterCount: inputFilterCount / 2
            )
            pool = AvgPool2D(poolSize: (2, 2), stride=2, padding="same")


        @differentiable
        member _.forward(input: Tensor) : Tensor (* <Float> *) {
            input.sequenced(through: conv, pool)



