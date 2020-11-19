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
    let pad = ZeroPadding2d(padding=((_padding, _padding), (_padding, _padding)))
    /// Convolution layer.
    let conv2d = Conv2d(inChannels, outChannels, kernelSize,strides = [stride; stride])

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    override _.forward(input: Tensor) : Tensor =
        input |> pad.forward |> conv2d.forward


type UNetSkipConnectionInnermost() = 
    inherit Model()
    let downConv: Conv2d
    let upConv: ConvTranspose2d
    let upNorm: BatchNorm<Float>
    
    public init(inChannels: int,
                innerChannels: int,
                outChannels: int) = 
        self.downConv = Conv2d(kernelSize=(4, 4, inChannels, innerChannels),
                               stride=2,
                               padding=kernelSize/2 (* "same " *),
                               filterInitializer: { dsharp.randn($0,
                                                            stddev=dsharp.scalar(0.02)))
        self.upNorm = BatchNorm2d(numFeatures=outChannels)
        
        self.upConv = ConvTranspose2d(kernelSize=(4, 4, innerChannels, outChannels),
                                       stride=2,
                                       padding=kernelSize/2 (* "same " *),
                                       filterInitializer: { dsharp.randn($0,
                                                                    stddev=dsharp.scalar(0.02)))

    
    
    override _.forward(input) =
        let x = leakyRelu(input)
        x = self.downConv(x)
        x = dsharp.relu(x)
        x = x |> self.upConv, self.upNorm)

        input.cat(x, alongAxis: 3)




type UNetSkipConnection<Sublayer: Layer>: Layer where Sublayer.TangentVector.VectorSpaceScalar = Float, Sublayer.Input = Tensor, Sublayer.Output =: Tensor =
    let downConv: Conv2d
    let downNorm: BatchNorm<Float>
    let upConv: ConvTranspose2d
    let upNorm: BatchNorm<Float>
    let dropOut = Dropout2d(p=0.5)
    let useDropOut: bool
    
    let submodule: Sublayer
    
    public init(inChannels: int,
                innerChannels: int,
                outChannels: int,
                submodule: Sublayer,
                useDropOut: bool = false) = 
        self.downConv = Conv2d(kernelSize=(4, 4, inChannels, innerChannels),
                               stride=2,
                               padding=kernelSize/2 (* "same " *),
                               filterInitializer: { dsharp.randn($0, stddev=dsharp.scalar(0.02)))
        self.downNorm = BatchNorm2d(numFeatures=innerChannels)
        self.upNorm = BatchNorm2d(numFeatures=outChannels)
        
        self.upConv = ConvTranspose2d(kernelSize=(4, 4, outChannels, innerChannels * 2),
                                       stride=2,
                                       padding=kernelSize/2 (* "same " *),
                                       filterInitializer: { dsharp.randn($0,
                                                                    stddev=dsharp.scalar(0.02)))
    
        self.submodule = submodule
        
        self.useDropOut = useDropOut

    
    
    override _.forward(input) =
        let x = leakyRelu(input)
        x = x |> self.downConv, self.downNorm, self.submodule)
        x = dsharp.relu(x)
        x = x |> self.upConv, self.upNorm)
        
        if self.useDropOut then
            x = self.dropOut(x)

        
        input.cat(x, alongAxis: 3)



type UNetSkipConnectionOutermost<Sublayer: Layer>: Layer where Sublayer.TangentVector.VectorSpaceScalar = Float, Sublayer.Input = Tensor, Sublayer.Output =: Tensor =
    let downConv: Conv2d
    let upConv: ConvTranspose2d
    
    let submodule: Sublayer
    
    public init(inChannels: int,
                innerChannels: int,
                outChannels: int,
                submodule: Sublayer) = 
        self.downConv = Conv2d(kernelSize=(4, 4, inChannels, innerChannels),
                               stride=2,
                               padding=kernelSize/2 (* "same " *),
                               filterInitializer: { dsharp.randn($0,
                                                            stddev=dsharp.scalar(0.02)))
        self.upConv = ConvTranspose2d(kernelSize=(4, 4, outChannels, innerChannels * 2),
                                       stride=2,
                                       padding=kernelSize/2 (* "same " *),
                                       activation= tanh,
                                       filterInitializer: { dsharp.randn($0,
                                                                    stddev=dsharp.scalar(0.02)))
    
        self.submodule = submodule

    
    
    override _.forward(input) =
        let x = input |> self.downConv, self.submodule)
        x = dsharp.relu(x)
        x = self.upConv(x)

        x


