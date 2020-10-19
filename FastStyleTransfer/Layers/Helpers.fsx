open DiffSharp

/// A 2-D layer applying padding with reflection over a mini-batch.
type ReflectionPad2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    type TangentVector = EmptyTangentVector

    /// The padding values along the spatial dimensions.
    @noDerivative let padding: ((Int, Int), (Int, Int))

    /// Creates a reflect-padding 2D Layer.
    ///
    /// - Parameter padding: A tuple of 2 tuples of two integers describing how many elements to
    ///   be padded at the beginning and end of each padding dimensions.
    public init(padding: ((Int, Int), (Int, Int))) = 
        self.padding = padding


    /// Creates a reflect-padding 2D Layer.
    ///
    /// - Parameter padding: Integer that describes how many elements to be padded
    ///   at the beginning and end of each padding dimensions.
    public init(padding: int) = 
        self.padding = ((padding, padding), (padding, padding))


    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer. Expected layout is BxHxWxC.
    /// - Returns: The output.
    @differentiable
    member _.forward(input: Tensor<Scalar>) : Tensor =
        // Padding applied to height and width dimensions only.
        return input.padded(forSizes: [
            (0, 0),
            padding.0,
            padding.1,
            (0, 0)
        ], mode: .reflect)



/// A layer applying `relu` activation function.
type ReLU<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    type TangentVector = EmptyTangentVector

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    @differentiable
    member _.forward(input: Tensor<Scalar>) : Tensor =
        return relu(input)


