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

#r @"..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#load "Layers.fsx"

open DiffSharp
open DiffSharp.Model
open Layers

type NetD(inChannels: int, lastConvFilters: int) =
    inherit Model()
    let kw = 4

    let model1 =
        Sequential [
            Conv2d(inChannels, lastConvFilters,kw, stride=2, padding=kw/2) // filterInitializer: { dsharp.randn($0, stddev=dsharp.scalar(0.02)))
            Function(dsharp.leakyRelu)

            Conv2d(lastConvFilters, 2 * lastConvFilters, stride=2, padding=kw/2) // filterInitializer: { dsharp.randn($0, stddev=dsharp.scalar(0.02)))
            BatchNorm2d(numFeatures=2 * lastConvFilters)
            Function(dsharp.leakyRelu)

            Conv2d(2 * lastConvFilters, 4 * lastConvFilters, kw, stride=2, padding=kw/2) //filterInitializer: { dsharp.randn($0, stddev=dsharp.scalar(0.02)))
            BatchNorm2d(numFeatures=4 * lastConvFilters)
            Function(dsharp.leakyRelu)
        ]

    let model2 =
        Sequential [
            model1
            ConvLayer(inChannels=4 * lastConvFilters, outChannels=8 * lastConvFilters, kernelSize=4, stride=1, padding=1)

            BatchNorm2d(numFeatures=8 * lastConvFilters)
            Function(dsharp.leakyRelu)

            ConvLayer(inChannels=8 * lastConvFilters, outChannels=1, kernelSize=4, stride=1, padding=1)
        ]
    
    override _.forward(input) =
        model2.forward(input)

