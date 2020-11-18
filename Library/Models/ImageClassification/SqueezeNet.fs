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
// SqueezeNet: AlexNet Level Accuracy with 50X Fewer Parameters
// Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally
// and Kurt Keutzer
// https://arxiv.org/pdf/1602.07360.pdf

type Fire() =
    inherit Model()
    let squeeze: Conv2D<Float>
    let expand1: Conv2D<Float>
    let expand3: Conv2D<Float>

    public init(
        inChannels=int,
        squeezeFilterCount: int,
        expand1FilterCount: int,
        expand3FilterCount: int
    ) = 
        squeeze = Conv2d(
            kernelSize=(1, 1, inputFilterCount, squeezeFilterCount),
            activation= dsharp.relu)
        expand1 = Conv2d(
            kernelSize=(1, 1, squeezeFilterCount, expand1FilterCount),
            activation= dsharp.relu)
        expand3 = Conv2d(
            kernelSize=(3, 3, squeezeFilterCount, expand3FilterCount),
            padding=kernelSize/2 (* "same " *),
            activation= dsharp.relu)


    
    override _.forward(input) =
        let squeezed = squeeze(input)
        let expanded1 = expand1(squeezed)
        let expanded3 = expand3(squeezed)
        expanded1.cat(expanded3, alongAxis: -1)



type SqueezeNetV1_0() =
    inherit Model()
    let conv1 = Conv2d(
        kernelSize=(7, 7, 3, 96),
        stride=2,
        padding=kernelSize/2 (* "same " *),
        activation= dsharp.relu)
    let maxPool1 = MaxPool2d(poolSize: (3, 3), stride=2)
    let fire2 = Fire(
        inChannels=96,
        squeezeFilterCount: 16,
        expand1FilterCount: 64,
        expand3FilterCount: 64)
    let fire3 = Fire(
        inChannels=128,
        squeezeFilterCount: 16,
        expand1FilterCount: 64,
        expand3FilterCount: 64)
    let fire4 = Fire(
        inChannels=128,
        squeezeFilterCount: 32,
        expand1FilterCount: 128,
        expand3FilterCount: 128)
    let maxPool4 = MaxPool2d(poolSize: (3, 3), stride=2)
    let fire5 = Fire(
        inChannels=256,
        squeezeFilterCount: 32,
        expand1FilterCount: 128,
        expand3FilterCount: 128)
    let fire6 = Fire(
        inChannels=256,
        squeezeFilterCount: 48,
        expand1FilterCount: 192,
        expand3FilterCount: 192)
    let fire7 = Fire(
        inChannels=384,
        squeezeFilterCount: 48,
        expand1FilterCount: 192,
        expand3FilterCount: 192)
    let fire8 = Fire(
        inChannels=384,
        squeezeFilterCount: 64,
        expand1FilterCount: 256,
        expand3FilterCount: 256)
    let maxPool8 = MaxPool2d(poolSize: (3, 3), stride=2)
    let fire9 = Fire(
        inChannels=512,
        squeezeFilterCount: 64,
        expand1FilterCount: 256,
        expand3FilterCount: 256)
    let conv10: Conv2D<Float>
    let avgPool10 = AvgPool2D<Float>(poolSize: (13, 13), stride=1)
    let dropout = Dropout2d(p=0.5)

    public init(classCount: int) = 
        conv10 = Conv2d(kernelSize=(1, 1, 512, classCount), stride=1, activation= dsharp.relu)


    
    override _.forward(input) =
        let convolved1 = input |> conv1, maxPool1)
        let fired1 = convolved1 |> fire2, fire3, fire4, maxPool4, fire5, fire6)
        let fired2 = fired1 |> fire7, fire8, maxPool8, fire9)
        let convolved2 = fired2 |> dropout, conv10, avgPool10)
            .view([input.shape.[0], conv10.filter.shape.[3]])
        convolved2



type SqueezeNetV1_1() =
    inherit Model()
    let conv1 = Conv2d(
        kernelSize=(3, 3, 3, 64),
        stride=2,
        padding=kernelSize/2 (* "same " *),
        activation= dsharp.relu)
    let maxPool1 = MaxPool2d(poolSize: (3, 3), stride=2)
    let fire2 = Fire(
        inChannels=64,
        squeezeFilterCount: 16,
        expand1FilterCount: 64,
        expand3FilterCount: 64)
    let fire3 = Fire(
        inChannels=128,
        squeezeFilterCount: 16,
        expand1FilterCount: 64,
        expand3FilterCount: 64)
    let maxPool3 = MaxPool2d(poolSize: (3, 3), stride=2)
    let fire4 = Fire(
        inChannels=128,
        squeezeFilterCount: 32,
        expand1FilterCount: 128,
        expand3FilterCount: 128)
    let fire5 = Fire(
        inChannels=256,
        squeezeFilterCount: 32,
        expand1FilterCount: 128,
        expand3FilterCount: 128)
    let maxPool5 = MaxPool2d(poolSize: (3, 3), stride=2)
    let fire6 = Fire(
        inChannels=256,
        squeezeFilterCount: 48,
        expand1FilterCount: 192,
        expand3FilterCount: 192)
    let fire7 = Fire(
        inChannels=384,
        squeezeFilterCount: 48,
        expand1FilterCount: 192,
        expand3FilterCount: 192)
    let fire8 = Fire(
        inChannels=384,
        squeezeFilterCount: 64,
        expand1FilterCount: 256,
        expand3FilterCount: 256)
    let fire9 = Fire(
        inChannels=512,
        squeezeFilterCount: 64,
        expand1FilterCount: 256,
        expand3FilterCount: 256)
    let conv10: Conv2D<Float>
    let avgPool10 = AvgPool2D<Float>(poolSize: (13, 13), stride=1)
    let dropout = Dropout2d(p=0.5)

    public init(classCount: int) = 
        conv10 = Conv2d(kernelSize=(1, 1, 512, classCount), stride=1, activation= dsharp.relu)


    
    override _.forward(input) =
        let convolved1 = input |> conv1, maxPool1)
        let fired1 = convolved1 |> fire2, fire3, maxPool3, fire4, fire5)
        let fired2 = fired1 |> maxPool5, fire6, fire7, fire8, fire9)
        let convolved2 = fired2 |> dropout, conv10, avgPool10)
            .view([input.shape.[0], conv10.filter.shape.[3]])
        convolved2


