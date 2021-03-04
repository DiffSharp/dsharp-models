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
#r @"..\bin\Debug\net5.0\publish\DiffSharp.Core.dll"
#r @"..\bin\Debug\net5.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\bin\Debug\net5.0\publish\Library.dll"

open DiffSharp
open DiffSharp.Model

type Identity() =
    inherit Model()
    override _.forward(input) =
        input

type InstanceNorm2d(featureCount: int, ?epsilon: Tensor) =
    inherit Model()

    /// Small value added in denominator for numerical stability.
    let epsilon = defaultArg epsilon (dsharp.scalar 1e-5)

    /// Learnable parameter scale for affine transformation.
    let scale = dsharp.ones [featureCount] |> Parameter

    /// Learnable parameter offset for affine transformation.
    let offset = dsharp.zeros [featureCount] |> Parameter

    do base.add([scale; offset])

    //[<ShapeCheck("N,100,H,W")>]
    override _.forward(input: Tensor) =

        // Calculate mean & variance along H,W axes.
        let mean = input.mean(dims=[2; 3])
        let variance = input.variance(dims=[2; 3])
        let norm = (input - mean) * dsharp.rsqrt(variance + epsilon)
        let res = norm * scale.value.view([featureCount;1;1]) + offset.value.view([featureCount;1;1])
        res

    override _.ToString() = sprintf "InstanceNorm2d(scale=%O, offset=%O, epsilon=%O)" scale offset epsilon


/// Creates 2D convolution with padding layer.
///
/// - Parameters:
///   - inChannels=Number of input channels in convolution kernel.
///   - outChannels: Number of output channels in convolution kernel.
///   - kernelSize: Convolution kernel size (both width and height).
///   - stride: Stride size (both width and height).
type ConvLayer(inChannels: int, outChannels: int, kernelSize: int, stride: int, ?padding: int) = 
    inherit Model()

    let _padding = defaultArg padding (kernelSize / 2)
    /// Padding layer.
    let pad = ZeroPadding2d(_padding, _padding)
    /// Convolution layer.
    let conv2d = Conv2d(inChannels, outChannels, kernelSize,strides = [stride; stride])

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    override _.forward(input: Tensor) : Tensor =
        input |> pad.forward |> conv2d.forward


type UNetSkipConnectionInnermost(inChannels: int, innerChannels: int, outChannels: int) = 
    inherit Model()
    let downConv = Conv2d(inChannels=inChannels, outChannels=innerChannels, kernelSize=4, stride=2, padding=2)
                            (* filterInitializer: { dsharp.randn($0,stddev=dsharp.scalar(0.02)) *) 
    let upNorm = BatchNorm2d(numFeatures=outChannels)
        
    let upConv = ConvTranspose2d(inChannels=innerChannels, outChannels=outChannels, kernelSize=4, stride=2, padding=2) (* ,
                                    filterInitializer: { dsharp.randn($0,stddev=dsharp.scalar(0.02)) *)

    override _.forward(input) =
        let x = input |> dsharp.leakyRelu
        let x = x |> downConv.forward
        let x = x |> dsharp.relu
        let x = x |> upConv.forward |> upNorm.forward
        input.cat(x, dim=3)

type UNetSkipConnection<'Sublayer when 'Sublayer :> Model>(inChannels: int,
                innerChannels: int,
                outChannels: int,
                submodule: 'Sublayer,
                ?useDropOut: bool) =
    inherit Model()
    let useDropOut = defaultArg useDropOut false
    let dropOut = Dropout2d(p=0.5)
    
    let downConv = Conv2d(inChannels, innerChannels, 4, stride=2, padding=2) (* filterInitializer: { dsharp.randn($0, stddev=dsharp.scalar(0.02)) *) 
    let downNorm = BatchNorm2d(numFeatures=innerChannels)
    let upNorm = BatchNorm2d(numFeatures=outChannels)
        
    let upConv = ConvTranspose2d(outChannels, innerChannels * 2, 4, stride=2, padding=2)
                                    //filterInitializer: { dsharp.randn($0, stddev=dsharp.scalar(0.02)))
    
    override _.forward(input) =
        let x = input |> dsharp.leakyRelu
        let x = x |> downConv.forward |> downNorm.forward |> submodule.forward
        let x = dsharp.relu(x)
        let x = x |> upConv.forward |> upNorm.forward
        
        let x = if useDropOut then dropOut.forward(x) else x
        
        input.cat(x, dim=3)

type UNetSkipConnectionOutermost<'Sublayer when 'Sublayer :> Model>(inChannels: int,
                innerChannels: int,
                outChannels: int,
                submodule: 'Sublayer) = 
    inherit Model()
    
    let downConv = Conv2d(inChannels, innerChannels, kernelSize=4, stride=2, padding=2)
                            //filterInitializer: { dsharp.randn($0,stddev=dsharp.scalar(0.02)))
    let upConv = ConvTranspose2d(outChannels, innerChannels * 2, kernelSize=4, stride=2, padding=2)
                                    //activation= tanh,
                                    //filterInitializer: { dsharp.randn($0,stddev=dsharp.scalar(0.02)))
    
    override _.forward(input) =
        let x = input |> downConv.forward |> submodule.forward
        let x = dsharp.relu x
        let x = upConv.forward x
        x
