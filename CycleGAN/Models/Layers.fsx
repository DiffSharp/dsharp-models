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

/// 2-D layer applying instance normalization over a mini-batch of inputs.
///
/// Creates a instance normalization 2D Layer.
///
/// - Parameters:
///   - featureCount: Size of the channel axis in the expected input.
///   - epsilon: Small scalar added for numerical stability.
///
/// Reference: [Instance Normalization](https://arxiv.org/abs/1607.08022)
[<ShapeCheck(100)>]
type InstanceNorm2D(featureCount: int, ?epsilon: Tensor) =
    inherit Model()
    
    /// Small value added in denominator for numerical stability.
    let epsilon = defaultArg epsilon (dsharp.scalar 1e-5)
    
    /// Learnable parameter scale for affine transformation.
    let scale = dsharp.ones [featureCount]  |> Parameter
    
    /// Learnable parameter offset for affine transformation.
    let offset = dsharp.zeros [featureCount] |> Parameter

    [<ShapeCheck("N,100,H,W")>]
    override _.forward(input: Tensor) =
    
        // Calculate mean & variance along H,W axes.
        let mean = input.mean(dims=[2; 3])
        let variance = input.variance(dims=[2; 3])
        let norm = (input - mean) * dsharp.rsqrt(variance + epsilon)
        let res = norm * scale.view([featureCount;1;1]) + offset.view([featureCount;1;1])
        res

    override _.ToString() = sprintf "InstanceNorm2D(scale=%O, offset=%O, epsilon=%O)" scale offset epsilon

[<ShapeCheck(100)>]
type ResNetBlock(channels: int, ?useDropOut: bool) =
    let useDropOut = defaultArg useDropOut false
    let conv1 = Conv2d(channels, channels, kernelSize=3, bias=true, padding=1 )
    let norm1 = InstanceNorm2D(featureCount=channels)

    let conv2 = Conv2d(channels, channels, kernelSize=3, bias=true, padding=1)
    let norm2 = InstanceNorm2D(featureCount=channels)

    let dropOut = Dropout(0.5)

    [<ShapeCheck("N,100,H,W")>]
    override _.forward(input: Tensor) =
        let retVal = input |> conv1.forward |> norm1.forward
        let retVal = retVal |> dsharp.relu

        let retVal = if useDropOut then retVal  |> dropOut.forward else retVal
        let retVal = retVal |> conv2.forward |> norm2.forward

        input + retVal

    override _.ToString() =
        sprintf "ResNetBlock(conv1=%O norm1=%O conv2=%O norm2=%O dropOut=%O)" conv1 norm1 conv2 norm2 dropOut

