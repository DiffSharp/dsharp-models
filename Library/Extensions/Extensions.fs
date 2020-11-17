namespace DiffSharp

open System
open System.Collections.Generic
open System.IO
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Util
open DiffSharp.ShapeChecking

[<AutoOpen>]
module DiffSharpExtensions =

    type FilePath = string
    type RandomNumberGenerator = System.Random
    let (</>) (a: FilePath) (b: string) : FilePath = Path.Combine(a,b)
    type Tensor with 
        member t.ndims = t.dim

        member t.reshape(shape: seq<int>) = t.view(shape)

        member t.reshape(shape: seq<Int>) = t.view(shape)

        member t.reshape(shape: Shape) = t.view(shape)

        member t.sqr() = t * t

        member t.toInt32() = t.toScalar() |> Convert.ToInt32

        member t.toFloat32() = t.toScalar() |> Convert.ToSingle

        member t.gelu() : Tensor = failwith "TBD"

        member t.permute([<ParamArray>] permutation: int[]) : Tensor = failwith "TBD"

        member t.rsqrt() = 1.0 / t.sqrt()

        member t.argmax(dim: int) : Tensor = failwith "tbd"

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


    type Sequential([<ParamArray>] models: Model[]) =
        inherit Model()
        do base.add(Array.map box models)
        override _.forward(input) = 
            (input, models) ||> Array.fold (fun input m -> m.forward input)
        new (models: seq<Model>) = Sequential(Seq.toArrayQuick models)
    
    type Function(f: Tensor -> Tensor) =
        inherit Model()

        override _.ToString() = sprintf "Function()"

        override m.forward(value) = f value

    type Flatten() =
        inherit Model()

        override _.ToString() = sprintf "Flatten()"

        override m.forward(value) = value.flatten()

      

    type dsharp with 
        static member gelu(input: Tensor) = input.gelu()

        static member permute(input: Tensor, [<ParamArray>] permutation: int[]) = input.permute(permutation)

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

        //static member scalar(t: scalar, ?dtype, ?device, ?backend) = dsharp.full(1, t, ?dtype=dtype, ?device=device, ?backend=backend)

        static member randn(shape: seq<int>, stddev: scalar, ?mean: scalar) = dsharp.randn(shape=shape)

        static member sigmoidCrossEntropy(logits:Tensor, labels:Tensor, ?reduction:string) = logits.oneLike()

    type Model with 
        member m.grad(input, loss) = 
            m.reverseDiff()
            dsharp.grad (fun t -> loss (m.forward t)) input
        member m.gradv(input, loss) = 
            m.reverseDiff()
            dsharp.gradv (fun t -> loss (m.forward t)) input

    type ZeroPadding2D(padding: (int*int) * (int * int)) =
       inherit Model()
       override m.forward(value) = value // TBD

    type UpSampling2D(size: int) =
       inherit Model()
       override m.forward(value) = value // TBD

    type LayerNorm(featureCount: int, axis: int) =
       inherit Model()
       let p_offset = Parameter(dsharp.zero())
       let p_scale = Parameter(dsharp.zero())
       member _.offset = p_offset
       member _.scale = p_scale
       override m.forward(value) = value // TBD

    type MaxPool2d(kernelSize: int, stride: int) =
       inherit Model()
       override m.forward(value) = value // TBD

    type RMSProp(model: Model, ?learningRate: Tensor, ?decay: double) =
        inherit ModelOptimizer(model)
        let learningRate = defaultArg learningRate (dsharp.tensor(1e-3))
        /// <summary>TBD</summary>
        override o.updateRule name t = failwith "tbd"

    type AdaDelta(model: Model) =
        inherit ModelOptimizer(model)
        /// <summary>TBD</summary>
        override o.updateRule name t = failwith "tbd"

    type ParameterGroupOptimizer() =
        inherit Optimizer()
        /// <summary>TBD</summary>
        override o.updateRule name t = failwith "tbd"

    let scalar (x: scalar) : scalar = x

    type Embedding(vocabularySize: int, embeddingSize: int, embeddingsInitializer: Shape -> Tensor) =
        let mutable t = Dictionary<Tensor, Tensor>()
        member _.Item with get (v: Tensor) = t.[v]
        member _.embeddings 
             with get() = 
                 t |> Seq.map (fun (KeyValue(a,b)) -> dsharp.stack [a;b]) |> dsharp.stack
             and set(vs: Tensor) = 
                 t <- 
                     vs.unstack() 
                     |> Array.map (fun ab -> match ab.unstack() with [| a; b |] -> KeyValuePair(a,b) | _ -> failwith "expected pair")
                     |> Dictionary

    let truncatedNormalInitializer(standardDeviation: Tensor) : Shape -> Tensor = failwith "truncatedNormalInitializer"

    type TangentVector(x:obj) = class end
