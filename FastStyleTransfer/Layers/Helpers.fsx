#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"

open DiffSharp
open DiffSharp.Model

/// Creates a reflect-padding 2D Layer.
///
/// A 2-D layer applying padding with reflection over a mini-batch.
///
/// - Parameter padding: A tuple of 2 tuples of two integers describing how many elements to
///   be padded at the beginning and end of each padding dimensions.
type ReflectionPad2D(p1: (int * int), p2: (int * int)) =
    inherit Model()

    /// Creates a reflect-padding 2D Layer.
    ///
    /// - Parameter padding: Integer that describes how many elements to be padded
    ///   at the beginning and end of each padding dimensions.
    new (padding: int) = ReflectionPad2D((padding, padding), (padding, padding))

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer. Expected layout is BxHxWxC.
    /// - Returns: The output.
    
    override _.forward(input: Tensor) : Tensor =
        // Padding applied to height and width dimensions only.
        input.pad(forSizes=[(0, 0); p1; p2; (0, 0) ], mode= ".reflect")

/// A layer applying `relu` activation function.
type ReLU() =
    inherit Model()

    /// Returns the output obtained from applying the layer to the given input.
    ///
    /// - Parameter input: The input to the layer.
    /// - Returns: The output.
    override _.forward(input: Tensor) : Tensor =
        dsharp.relu(input)


