#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#load @"..\Layers\Normalization.fsx"
#load @"..\Layers\Helpers.fsx"

open DiffSharp
open DiffSharp.Model
open Normalization

/// Creates 2D residual block layer.
///
/// - Parameter channels: Number of input channels in convolution kernel.
type ResidualBlock(channels: int) = 
    inherit Model()
    /// Convolution & instance normalization layers.
    let conv1 = ConvLayer(inChannels=channels, outChannels=channels, kernelSize=3, stride=1)
    let in1 = InstanceNorm2d(featureCount=channels)
    let conv2 = ConvLayer(inChannels=channels, outChannels=channels, kernelSize=3, stride=1)
    let in2 = InstanceNorm2d(featureCount=channels)

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    override _.forward(input) =
        input + input |> conv1, in1, relu, conv2, in2)

/// Creates 2D upsampling layer.
///
/// Helper layer for upsampling.
///
/// Upsamples the input and then does a convolution.
/// Reference: http://distill.pub/2016/deconv-checkerboard/
///
/// - Parameters:
///   - inChannels=Number of input channels in convolution kernel.
///   - outChannels: Number of output channels in convolution kernel.
///   - kernelSize: Convolution kernel size (both width and height).
///   - stride: Stride size (both width and height).
///   - scaleFactor: Scale factor.
type UpsampleConvLayer(inChannels: int, outChannels: int, kernelSize: int, stride: int, ?scaleFactor: double) =
    inherit Model()
    /// Scale factor.
    let scaleFactor = defaultArg scaleFactor 1.0
    /// Padding layer.
    let reflectionPad = ReflectionPad2D<Float>(padding: int(kernelSize / 2))
    /// Convolution layer.
    let conv2d = Conv2d(kernelSize=(kernelSize, kernelSize, inChannels, outChannels),strides = [stride; stride])

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    
    override _.forward(input) =
        let newHeight = int(roundf(double(input.shape.[input.ndims - 3]) * scaleFactor))
        let newWidth = int(roundf(double(input.shape.[input.ndims - 2]) * scaleFactor))
        let resizedInput = resize(images=input, size= (newHeight, newWidth), method=".nearest")
        resizedInput |> reflectionPad, conv2d


/// A model that applies style.
type TransformerNet() = 
    inherit Model()
    // Convolution & instance normalization layers.
    let conv1 = ConvLayer(inChannels=3, outChannels: 32, kernelSize: 9, stride=1)
    let in1 = InstanceNorm2d(numFeatures=32)
    let conv2 = ConvLayer(inChannels=32, outChannels: 64, kernelSize=3, stride=2)
    let in2 = InstanceNorm2d(numFeatures=64)
    let conv3 = ConvLayer(inChannels=64, outChannels: 128, kernelSize=3, stride=2)
    let in3 = InstanceNorm2d(numFeatures=128)

    // Residual block layers.
    let res1 = ResidualBlock(channels: 128)
    let res2 = ResidualBlock(channels: 128)
    let res3 = ResidualBlock(channels: 128)
    let res4 = ResidualBlock(channels: 128)
    let res5 = ResidualBlock(channels: 128)

    // Upsampling & instance normalization layers.
    let deconv1 = UpsampleConvLayer(
        inChannels=128, outChannels: 64,
        kernelSize=3, stride=1, scaleFactor: 2.0)
    let in4 = InstanceNorm2d(numFeatures=64)
    let deconv2 = UpsampleConvLayer(
        inChannels=64, outChannels: 32,
        kernelSize=3, stride=1, scaleFactor: 2.0)
    let in5 = InstanceNorm2d(numFeatures=32)
    let deconv3 = UpsampleConvLayer(
        inChannels=32, outChannels: 3,
        kernelSize: 9, stride=1)

    // ReLU activation layer.
    let relu = ReLU<Float>()

    /// Creates style transformer model.
    public init() =

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer. Expected layout is BxHxWxC.
    /// - Returns: The output.
    
    override _.forward(input) =
        let convolved1 = input |> conv1, in1, relu)
        let convolved2 = convolved1 |> conv2, in2, relu)
        let convolved3 = convolved2 |> conv3, in3, relu)
        let residual = convolved3 |> res1, res2, res3, res4, res5)
        let upscaled1 = residual |> deconv1, in4, relu)
        let upscaled2 = upscaled1 |> deconv2, in5)
        let upscaled3 = deconv3(upscaled2)
        upscaled3

/// Helper layer: convolution with padding.
/// Creates 2D convolution with padding layer.
///
/// - Parameters:
///   - inChannels=Number of input channels in convolution kernel.
///   - outChannels: Number of output channels in convolution kernel.
///   - kernelSize: Convolution kernel size (both width and height).
///   - stride: Stride size (both width and height).
type ConvLayer(inChannels: int, outChannels: int, kernelSize: int, stride: int) = 
    inherit Model()

    /// Padding layer.
    let reflectionPad = ReflectionPad2D<Float>(padding=int(kernelSize / 2))
    /// Convolution layer.
    let conv2d = Conv2d(kernelSize=(kernelSize, kernelSize, inChannels, outChannels), strides = [stride, stride])

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    override _.forward(input) =
        input |> reflectionPad.forward |> conv2d.forward

