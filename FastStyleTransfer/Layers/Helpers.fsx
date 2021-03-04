#r @"..\..\bin\Debug\net5.0\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\net5.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\net5.0\publish\Library.dll"

open DiffSharp
open DiffSharp.Model

/// Creates a reflect-padding 2D Layer.
///
/// A 2-D layer applying padding with reflection over a mini-batch.
/// Padding applied to height and width dimensions only.
type ReflectionPad2D(padding: int) =
    inherit Model()

    override _.forward(input: Tensor) : Tensor =
        input.pad([0;0;padding;padding])

/// A layer applying `relu` activation function.
type ReLU() =
    inherit Model()

    override _.forward(input: Tensor) : Tensor =
/// Returns the output obtained from applying the layer to the given input.
        dsharp.relu(input)
