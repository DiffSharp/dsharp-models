open DiffSharp

/// 2-D layer applying instance normalization over a mini-batch of inputs.
///
/// Reference: [Instance Normalization](https://arxiv.org/abs/1607.08022)
type InstanceNorm2D<Scalar: TensorFlowFloatingPoint>: Layer {
    /// Learnable parameter scale for affine transformation.
    let scale: Tensor<Scalar>
    /// Learnable parameter offset for affine transformation.
    let offset: Tensor<Scalar>
    /// Small value added in denominator for numerical stability.
    let epsilon: Tensor<Scalar>

    /// Creates a instance normalization 2D Layer.
    ///
    /// - Parameters:
    ///   - featureCount: Size of the channel axis in the expected input.
    ///   - epsilon: Small scalar added for numerical stability.
    public init(featureCount=int, epsilon: Tensor<Scalar> = dsharp.tensor(1e-5)) = 
        self.epsilon = epsilon
        scale = Tensor<Scalar>(ones: [featureCount])
        offset = Tensor<Scalar>(zeros: [featureCount])


    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer. Expected input layout is BxHxWxC.
    /// - Returns: The output.
    
    override _.forward(input: Tensor<Scalar>) : Tensor =
        // Calculate mean & variance along H,W axes.
        let mean = input.mean(dim=[1; 2])
        let variance = input.variance(dim=[1; 2])
        let norm = (input - mean) * rsqrt(variance + epsilon)
        return norm * scale + offset


