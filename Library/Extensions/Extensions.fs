namespace DiffSharp

open System.IO
open DiffSharp.Model
open DiffSharp.Util
open DiffSharp.ShapeChecking
open System

[<AutoOpen>]
module DiffSharpExtensions =

    type FilePath = string
    type RandomNumberGenerator = System.Random
    let (</>) (a: FilePath) (b: string) : FilePath = Path.Combine(a,b)
    type Tensor with 
        member x.ndims = x.dim
        member x.reshape(shape: seq<int>) = x.view(shape)
        member x.reshape(shape: seq<Int>) = x.view(shape)
        member x.sqr() = x * x
        member x.toInt32() = x.toScalar() |> Convert.ToInt32
        member x.toFloat32() = x.toScalar() |> Convert.ToSingle

    type dsharp with 
        static member scalar(x: scalar, ?dtype, ?device, ?backend) = dsharp.full(1, x, ?dtype=dtype, ?device=device, ?backend=backend)
        static member randn(shape: seq<int>, stddev: scalar, ?mean: scalar) = dsharp.randn(shape=shape)
        static member sigmoidCrossEntropy(logits: Tensor, labels: Tensor, ?reduction: string) = logits.oneLike()

    type Dense(inputSize: Int, outputSize: Int, ?activation: (Tensor -> Tensor)) =
        inherit Model()
        override _.forward(input) = failwith ""
        new (inputSize: int, outputSize: int, ?activation: (Tensor -> Tensor)) =
              Dense(Int inputSize, Int outputSize, ?activation=activation)

    type Sequential([<ParamArray>] models: Model[]) =
        inherit Model()
        override _.forward(input) = 
            (input, models) ||> Array.fold (fun input m -> m.forward input)
    
    type Function(f: Tensor -> Tensor) =
        inherit Model()

        override _.ToString() = sprintf "Function()"

        override m.forward(value) = f value

    type Flatten() =
        inherit Model()

        override _.ToString() = sprintf "Flatten()"

        override m.forward(value) = value.flatten()

        
    type Tensor with 
        member t.rsqrt() = 1.0 / t.sqrt()

        member t.unsqueeze (dims: seq<int>) =
            let dims = dims |> Seq.toArrayQuick
            (t, Array.rev (Array.sort dims)) ||> Array.fold (fun input dim -> dsharp.unsqueeze(input, dim))

        member t.squeeze (dims: seq<int>) =
            let dims = dims |> Seq.toArrayQuick
            (t, Array.rev (Array.sort dims)) ||> Array.fold (fun input dim -> dsharp.squeeze(input, dim))

        member t.mean (dims: seq<int>, ?keepDim: bool) =
            let dims = dims |> Seq.toArrayQuick
            (t, Array.rev (Array.sort dims)) ||> Array.fold (fun input dim -> dsharp.mean(input, dim, ?keepDim=keepDim))

        member t.variance (dims: seq<int>, ?keepDim: bool) =
            let dims = dims |> Seq.toArrayQuick
            (t, Array.rev (Array.sort dims)) ||> Array.fold (fun input dim -> dsharp.variance(input, dim, ?keepDim=keepDim))

        member t.sum (dims: seq<int>, ?keepDim: bool) =
            let dims = dims |> Seq.toArrayQuick
            (t, Array.rev (Array.sort dims)) ||> Array.fold (fun input dim -> dsharp.sum(input, dim, ?keepDim=keepDim))

        member t.stddev (dims: seq<int>, ?keepDim: bool) =
            let dims = dims |> Seq.toArrayQuick
            (t, Array.rev (Array.sort dims)) ||> Array.fold (fun input dim -> dsharp.stddev(input, dim, ?keepDim=keepDim))

        member t.moments () =
               dsharp.mean(t), dsharp.stddev(t)

        member t.moments (dim: int, ?keepDim: bool) =
               t.mean(dim, ?keepDim=keepDim), t.stddev(dim, ?keepDim=keepDim)

        member t.moments (dims: seq<int>, ?keepDim: bool) =
               t.mean(dims, ?keepDim=keepDim), t.stddev(dims, ?keepDim=keepDim)


    type dsharp with 
        static member rsqrt(input: Tensor) = input.rsqrt()

        static member squeeze (input: Tensor, dims: seq<int>) = input.squeeze(dims)

        static member unsqueeze (input: Tensor, dims: seq<int>) = input.unsqueeze(dims)

        static member mean (input: Tensor, dims: seq<int>, ?keepDim: bool) = input.mean(dims, ?keepDim=keepDim)

        static member variance (input: Tensor, dims: seq<int>, ?keepDim: bool) = input.variance(dims, ?keepDim=keepDim)

        static member sum (input: Tensor, dims: seq<int>, ?keepDim: bool) = input.sum(dims, ?keepDim=keepDim)

        static member stddev (input: Tensor, dims: seq<int>, ?keepDim: bool) = input.stddev(dims, ?keepDim=keepDim)

        static member moments (input: Tensor) = input.moments()

        static member moments (input: Tensor, dim: int, ?keepDim: bool) = input.moments(dim, ?keepDim=keepDim)

        static member moments (input: Tensor, dims: seq<int>, ?keepDim: bool) = input.moments(dims, ?keepDim=keepDim)

        static member depthwiseConv2d(input:Tensor, filters:Tensor, ?stride:int, ?strides:seq<int>, ?padding:int, ?paddings:seq<int>, ?dilation:int, ?dilations:seq<int>) =
            // TODO: see https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843
            // needs "groups"
            input.conv2d(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
