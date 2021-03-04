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
#r @"..\..\bin\Debug\net5.0\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\net5.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\net5.0\publish\Library.dll"

#load "Layers.fsx"
open DiffSharp
open DiffSharp.Model
open DiffSharp.ShapeChecking
open Layers

[<ShapeCheck(3, 3, 5, 7)>]
type ResNetGenerator(inChannels:int,
        outChannels: int,
        blocks: int,
        ngf: int,
        ?useDropout: bool) =
    inherit Model()

    let useDropout = defaultArg useDropout false
    let useBias = true

    let conv1 = Conv2d(inChannels, ngf, kernelSize=7, stride=1, bias=useBias)
    let norm1 = InstanceNorm2d(numFeatures=ngf)

    let mult = 1

    let conv2 = Conv2d(ngf * mult, ngf * mult * 2, kernelSize=3, stride=2, padding = 1, bias=useBias)
    let norm2 = InstanceNorm2d(numFeatures=ngf * mult * 2)

    let mult = 2

    let conv3 = Conv2d(ngf * mult, ngf * mult * 2, kernelSize=3, stride=2, padding = 1, bias=useBias)
    let norm3 = InstanceNorm2d(numFeatures=ngf * mult * 2)

    let mult = 4

    let resblocks = Array.init blocks (fun _ ->  ResNetBlock(channels=ngf * mult, useDropOut=useDropout))

    let mult = 4

    let upConv1 = ConvTranspose2d(ngf * mult, ngf * mult / 2, kernelSize=3, stride=2, padding=1, outputPadding=1, bias=useBias)
    let upNorm1 = InstanceNorm2d(numFeatures=ngf * mult / 2)
    
    let mult = 2

    let upConv2 = ConvTranspose2d(ngf * mult, ngf * mult / 2, kernelSize=3, stride=2, padding=1, outputPadding=1, bias=useBias)
    let upNorm2 = InstanceNorm2d(numFeatures=ngf * mult / 2)

    let lastConv = Conv2d(ngf, outChannels, kernelSize=7, bias=useBias, padding=3)

    do base.register()

    [<ShapeCheck("N, 3, 748, 748")>]
    override _.forward(input: Tensor) = 
        let x = input.pad([0;0;3;3])
        let x = x |> conv1.forward |> norm1.forward |> dsharp.relu
        let x = x |> conv2.forward |> norm2.forward |> dsharp.relu
        let x = x |> conv3.forward |> norm3.forward |> dsharp.relu

        let x = (x, resblocks) ||> Array.fold (fun x resblock -> resblock.forward x)

        let x = x |> upConv1.forward |> upNorm1.forward |> dsharp.relu
        let x = x |> upConv2.forward |> upNorm2.forward |> dsharp.relu

        let x = x |> lastConv.forward
        let x = tanh(x)

        x


