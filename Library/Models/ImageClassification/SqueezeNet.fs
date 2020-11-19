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

module Models.ImageClassification.SqueezeNet

open DiffSharp
open DiffSharp.Model

// Original Paper:
// SqueezeNet: AlexNet Level Accuracy with 50X Fewer Parameters
// Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally
// and Kurt Keutzer
// https://arxiv.org/pdf/1602.07360.pdf

type Fire(inputChannels:int,
        squeezeChannels: int,
        expand1Channels: int,
        expand3Channels: int) =
    inherit Model()
    let squeeze = Conv2d(inputChannels, squeezeChannels, kernelSize=1, activation=dsharp.relu)
    let expand1 = Conv2d(squeezeChannels, expand1Channels, kernelSize=1, activation=dsharp.relu)
    let expand3 = Conv2d(squeezeChannels, expand3Channels, kernelSize=3, padding=1 (* "same " *),activation=dsharp.relu)

    override _.forward(input) =
        let squeezed = squeeze.forward(input)
        let expanded1 = expand1.forward(squeezed)
        let expanded3 = expand3.forward(squeezed)
        expanded1.cat(expanded3, dim= -1)

type SqueezeNetV1_0(classCount: int) =
    inherit Model()
    let conv1 = Conv2d(3, 96, kernelSize=7, stride=2, padding=3 (* "same " *), activation=dsharp.relu)
    let maxPool1 = MaxPool2d(kernelSize=3, stride=2)
    let fire2 =
        Fire(inputChannels=96,
            squeezeChannels=16,
            expand1Channels=64,
            expand3Channels=64)
    let fire3 = 
        Fire(inputChannels=128,
            squeezeChannels=16,
            expand1Channels=64,
            expand3Channels=64)
    let fire4 =
        Fire(inputChannels=128,
            squeezeChannels=32,
            expand1Channels=128,
            expand3Channels=128)
    let maxPool4 = MaxPool2d(kernelSize=3, stride=2)
    let fire5 =
        Fire(inputChannels=256,
            squeezeChannels=32,
            expand1Channels=128,
            expand3Channels=128)
    let fire6 =
        Fire(inputChannels=256,
            squeezeChannels=48,
            expand1Channels=192,
            expand3Channels=192)
    let fire7 =
        Fire(inputChannels=384,
            squeezeChannels=48,
            expand1Channels=192,
            expand3Channels=192)
    let fire8 =
        Fire(inputChannels=384,
            squeezeChannels=64,
            expand1Channels=256,
            expand3Channels=256)
    let maxPool8 = MaxPool2d(kernelSize=3, stride=2)
    let fire9 =
        Fire(inputChannels=512,
            squeezeChannels=64,
            expand1Channels=256,
            expand3Channels=256)
    let avgPool10 = AvgPool2d(kernelSize=13, stride=1)
    let dropout = Dropout2d(p=0.5)

    let conv10 = Conv2d(512, classCount, kernelSize=1, stride=1, activation=dsharp.relu)
    
    override _.forward(input) =
        let convolved1 = input |> conv1.forward |> maxPool1.forward
        let fired1 = convolved1 |> fire2.forward |> fire3.forward |> fire4.forward |> maxPool4.forward |> fire5.forward |> fire6.forward
        let fired2 = fired1 |> fire7.forward |> fire8.forward |> maxPool8.forward |> fire9.forward
        let convolved2 = fired2 |> dropout.forward |> conv10.forward |> avgPool10.forward
        let convolved2 = convolved2.view([input.shape.[0]; failwith "tbd - check me" (* conv10.filter.shape.[3] *)])
        convolved2

type SqueezeNetV1_1(classCount: int) =
    inherit Model()
    let conv1 =
        Conv2d(3, 64, 
            kernelSize=3,
            stride=2,
            padding=1 (* "same " *),
            activation=dsharp.relu)
    let maxPool1 = MaxPool2d(kernelSize=3, stride=2)
    let fire2 =
        Fire(inputChannels=64,
            squeezeChannels=16,
            expand1Channels=64,
            expand3Channels=64)
    let fire3 = 
        Fire(inputChannels=128,
            squeezeChannels=16,
            expand1Channels=64,
            expand3Channels=64)
    let maxPool3 = MaxPool2d(kernelSize=3, stride=2)
    let fire4 =
        Fire(inputChannels=128,
            squeezeChannels=32,
            expand1Channels=128,
            expand3Channels=128)
    let fire5 =
        Fire(inputChannels=256,
            squeezeChannels=32,
            expand1Channels=128,
            expand3Channels=128)
    let maxPool5 = MaxPool2d(kernelSize=3, stride=2)
    let fire6 =
        Fire(inputChannels=256,
            squeezeChannels=48,
            expand1Channels=192,
            expand3Channels=192)
    let fire7 =
        Fire(inputChannels=384,
            squeezeChannels=48,
            expand1Channels=192,
            expand3Channels=192)
    let fire8 =
        Fire(inputChannels=384,
            squeezeChannels=64,
            expand1Channels=256,
            expand3Channels=256)
    let fire9 =
        Fire(inputChannels=512,
            squeezeChannels=64,
            expand1Channels=256,
            expand3Channels=256)
    let conv10 = Conv2d(512, classCount, kernelSize=1, stride=1, activation=dsharp.relu)
    let avgPool10 = AvgPool2d(kernelSize=13, stride=1)
    let dropout = Dropout2d(p=0.5)

    override _.forward(input) =
        let convolved1 = input |> conv1.forward |> maxPool1.forward
        let fired1 = convolved1 |> fire2.forward |> fire3.forward |> maxPool3.forward |> fire4.forward |> fire5.forward
        let fired2 = fired1 |> maxPool5.forward |> fire6.forward |> fire7.forward |> fire8.forward |> fire9.forward
        let convolved2 = fired2 |> dropout.forward |> conv10.forward |> avgPool10.forward
        let convolved2 = convolved2.view([input.shape.[0]; failwith "tbd check me" (* conv10.filter.shape.[3] *) ])
        convolved2
