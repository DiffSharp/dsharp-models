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
#r @"..\..\bin\Debug\netcoreapp3.0\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.0\publish\Library.dll"

open DiffSharp
open DiffSharp.Model
open DiffSharp.ShapeChecking

[<ShapeCheck(3, "K")>]
type NetD(inChannels: Int, lastConvFilters: Int) =
    inherit Model()

    let kw = 4I

    let module1 = 
        Sequential(
            Function (fun t -> t),
            Conv2d(inChannels, lastConvFilters, kernelSize=kw, stride=2I, padding=kw/2I),
            Function dsharp.leakyRelu,

            Function (fun t -> t),
            Conv2d(lastConvFilters, 2*lastConvFilters, kernelSize=kw, stride=2I, padding=kw/2I),
            Function (fun t -> t),
            BatchNorm2d(numFeatures=2*lastConvFilters),
            Function (fun t -> t),
            Function dsharp.leakyRelu,

            Function (fun t -> t),
            Conv2d(2*lastConvFilters, 4*lastConvFilters, kernelSize=kw, stride=2I, padding=kw/2I),
            Function (fun t -> t),
            BatchNorm2d(numFeatures=4*lastConvFilters),
            Function (fun t -> t),
            Function dsharp.leakyRelu
        )

    let module2 = 
        Sequential (
            module1,
            Conv2d(inChannels=4*lastConvFilters, outChannels=8*lastConvFilters, kernelSize=4I, stride=1I, padding=1I),
            BatchNorm2d(numFeatures=8*lastConvFilters),
            Function dsharp.leakyRelu,
            
            Conv2d(inChannels=8*lastConvFilters, outChannels=1I, kernelSize=4I, stride=1I, padding=1I),
            Function (fun t -> t)
        )

    [<ShapeCheck("N,3,748,748", ReturnShape="N,1,93,93")>]
    override _.forward(input: Tensor) = 
        module2.forward(input)


