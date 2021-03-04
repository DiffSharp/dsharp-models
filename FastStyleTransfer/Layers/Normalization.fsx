#r @"..\..\bin\Debug\net5.0\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\net5.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\net5.0\publish\Library.dll"

open DiffSharp
open DiffSharp.Model

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
