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
// "Very Deep Convolutional Networks for Large-Scale Image Recognition"
// Karen Simonyan, Andrew Zisserman
// https://arxiv.org/abs/1409.1556

type VGGBlock() =
    inherit Model()
    let blocks: [Conv2d] = []
    let maxpool = MaxPool2d(kernelSize=2, stride=2)

    public init(featureCounts: (Int, Int, Int, Int), blockCount: int) = 
        self.blocks = [Conv2d(kernelSize=(3, 3, featureCounts.0, featureCounts.1),
            padding=kernelSize/2 (* "same " *),
            activation= dsharp.relu)]
        for _ in 1..blockCount-1 do
            self.blocks <- blocks + [Conv2d(kernelSize=(3, 3, featureCounts.2, featureCounts.3),
                padding=kernelSize/2 (* "same " *),
                activation= dsharp.relu)]



    
    override _.forward(input) =
        maxpool(blocks.differentiableReduce(input) =  $1($0))



type VGG16() =
    inherit Model()
    let layer1: VGGBlock
    let layer2: VGGBlock
    let layer3: VGGBlock
    let layer4: VGGBlock
    let layer5: VGGBlock

    let flatten = Flatten()
    let dense1 = Linear(inFeatures=512 * 7 * 7, outFeatures=4096, activation= dsharp.relu)
    let dense2 = Linear(inFeatures=4096, outFeatures=4096, activation= dsharp.relu)
    let output: Dense

    public init(classCount: int = 1000) = 
        layer1 = VGGBlock(featureCounts: (3, 64, 64, 64), blockCount=2)
        layer2 = VGGBlock(featureCounts: (64, 128, 128, 128), blockCount=2)
        layer3 = VGGBlock(featureCounts: (128, 256, 256, 256), blockCount=3)
        layer4 = VGGBlock(featureCounts: (256, 512, 512, 512), blockCount=3)
        layer5 = VGGBlock(featureCounts: (512, 512, 512, 512), blockCount=3)
        output = Linear(inFeatures=4096, outFeatures=classCount)


    
    override _.forward(input) =
        let backbone = input |> layer1, layer2, layer3, layer4, layer5)
        backbone |> flatten, dense1, dense2, output)



type VGG19() =
    inherit Model()
    let layer1: VGGBlock
    let layer2: VGGBlock
    let layer3: VGGBlock
    let layer4: VGGBlock
    let layer5: VGGBlock

    let flatten = Flatten()
    let dense1 = Linear(inFeatures=512 * 7 * 7, outFeatures=4096, activation= dsharp.relu)
    let dense2 = Linear(inFeatures=4096, outFeatures=4096, activation= dsharp.relu)
    let output: Dense

    public init(classCount: int = 1000) = 
        layer1 = VGGBlock(featureCounts: (3, 64, 64, 64), blockCount=2)
        layer2 = VGGBlock(featureCounts: (64, 128, 128, 128), blockCount=2)
        layer3 = VGGBlock(featureCounts: (128, 256, 256, 256), blockCount=4)
        layer4 = VGGBlock(featureCounts: (256, 512, 512, 512), blockCount=4)
        layer5 = VGGBlock(featureCounts: (512, 512, 512, 512), blockCount=4)
        output = Linear(inFeatures=4096, outFeatures=classCount)


    
    override _.forward(input) =
        let backbone = input |> layer1, layer2, layer3, layer4, layer5)
        backbone |> flatten, dense1, dense2, output)


