namespace DiffSharp

open System
open System.Collections.Generic
open System.IO
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Util
open DiffSharp.ShapeChecking
open System.Runtime.CompilerServices

[<AutoOpen>]
module DiffSharpExtensions =

    type FilePath = string
    type RandomNumberGenerator = System.Random

    let (</>) (a: FilePath) (b: string) : FilePath = Path.Combine(a,b)

    let scalar (x: scalar) : scalar = x

    type Tensor with 
        member t.ndims = t.dim

        member t.reshape(shape: seq<int>) = t.view(shape)

        member t.reshape(shape: seq<Int>) = t.view(shape)

        member t.reshape(shape: Shape) = t.view(shape)

        member t.cat(t2: Tensor, dim: int) : Tensor =
            dsharp.cat([t;t2], dim)
            //t.split(([|  |]: int[]), dim=dim)

        member t.sqr() = t * t

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

        member t.sequenced([<ParamArray>] models: Model[] ) =
            let mutable res = t
            for m in models do
                res <- m.forward res
            res

        member t.argmax(dim: int) : Tensor = failwith "TBD - argmax along dimension"

        member a.chunk(count: int, ?dim: int) =
            let dim = defaultArg dim 0
            let n = a.shape.[dim]
            let n = 5
            let count = 3
            let sz = (n + count - 1) / count
            let sizes = [| for i in 0 .. count-1 do let k = min (n - i * sz) sz in if k > 0 then yield k |]
            a.split(sizes, dim=dim)

    type Sequential([<ParamArray>] models: Model[]) =
        inherit Model()
        do base.add(Array.map box models)
        override _.forward(input) = 
            (input, models) ||> Array.fold (fun input m -> m.forward input)
        new (models: seq<Model>) = Sequential(Seq.toArrayQuick models)
    
    type Function(f: Tensor -> Tensor) =
        inherit Model()

        override _.ToString() = sprintf "Function(%O)" f

        override m.forward(value) = f value

    [<AutoOpen>]
    type Functions =
        static member ZeroPadding2d(p0: int, p1: int) =
           Function (fun value -> value.pad([0; 0; p0; p1]))

        static member AvgPool2d(kernelSize: int, ?stride: int, ?padding: int) =
            Function (fun input -> input.avgpool2d(kernelSize, ?stride=stride, ?padding=padding))

        static member GlobalAvgPool2d() =
            // See https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/18
            Function (fun x -> dsharp.mean(x.view([ x.shapex.[0]; x.shapex.[1]; -1I]), dim=2))
           
        static member Flatten() =
            Function (fun value -> value.flatten(startDim=(if value.dim=1 then 0 else 1)))

        static member MaxPool2d(?kernelSize: int, ?stride: int, ?padding: int, ?kernelSizes: seq<int>, ?strides: seq<int>, ?paddings: seq<int>) =
           Function (fun value -> value.maxpool2d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings))

        static member MaxPool3d(?kernelSize: int, ?stride: int, ?padding: int, ?kernelSizes: seq<int>, ?strides: seq<int>, ?paddings: seq<int>) =
           Function (fun value -> value.maxpool3d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings))

    type dsharp with 
        static member gelu(input: Tensor) = input.gelu()

        static member permute(input: Tensor, [<ParamArray>] permutation: seq<int>) = input.permute(permutation)

        static member rsqrt(input: Tensor) = input.rsqrt()

        static member hardsigmoid(input: Tensor) = input.hardsigmoid()

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

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with the given mean and standard deviation.</summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="stddev">The desired standard deviation of returned tensor.</param>
        /// <param name="mean">The desired mean of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randn(shape: seq<int>, stddev: scalar, ?mean: scalar, ?dtype, ?device, ?backend) =
            let _backend = defaultArg backend Backend.Default
            let mean = defaultArg mean (0.0 :> scalar)
            dsharp.randn(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend) * stddev + mean

        static member rand(shape: seq<int>, low: scalar, high: scalar, ?dtype, ?device, ?backend) =
            dsharp.rand(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        static member sigmoidCrossEntropy(logits:Tensor, labels:Tensor, ?reduction:string) =
            if logits.backend = Backend.ShapeChecking then 
                logits.oneLike() 
            else 
                failwith "TBD"

        static member silu(input:Tensor) : Tensor = input.silu()

        static member hardswish(input:Tensor) : Tensor = input.hardswish()

        static member relu6(input:Tensor) : Tensor = input.relu6()

    [<Extension>]
    type ModelExtensions() =
        [<Extension>]
        static member grad(model: Model<Tensor, Tensor>, input, loss: Tensor -> Tensor) = 
            model.reverseDiff()
            let output = model.forward(input)
            (loss output).reverse()
            model.parameters.derivative

        [<Extension>]
        static member gradv(model: Model<Tensor, Tensor>, input, loss: Tensor -> Tensor) = 
            model.reverseDiff()
            let output = model.forward(input)
            (loss output).reverse()
            output, model.parameters.derivative
            //model.reverseDiff()
            //dsharp.gradv (fun t -> model.forwardLoss (fun a b -> dsharp.mseLoss(a,b)) input t model.parameters) input

        [<Extension>]
        static member appliedForBackpropagation(model: Model<Tensor, Tensor>, input) : Tensor * (Tensor -> Tensor * Tensor)= 
            failwith "tbd"
            //m.reverseDiff()
            //dsharp.gradv (fun t -> loss (m.forward t)) input

    type UpSampling2d(size: int) =
       inherit Model()
       override m.forward(value) = failwith "TBD"; value

    type LayerNorm(numFeatures: int, axis: int) =
       inherit Model()
       let p_offset = Parameter(dsharp.zero())
       let p_scale = Parameter(dsharp.zero())
       member _.offset = p_offset
       member _.scale = p_scale
       override m.forward(value) = failwith "TBD"; value

    type DepthwiseConv2d(inChannels: int, channelMultiplier: int, ?kernelSize: int, ?kernelSizes: seq<int>, ?stride: int, ?padding: int, ?strides: seq<int>, ?paddings: seq<int>) =
       inherit Model()
       // TODO this is fake
       let fake_conv2d = Conv2d(inChannels=inChannels, outChannels=(inChannels*channelMultiplier), ?kernelSize=kernelSize, ?kernelSizes=kernelSizes, ?stride=stride, ?padding=padding, ?strides=strides, ?paddings=paddings) //failwith "TBD"; value
       override m.forward(value) = fake_conv2d.forward(value)

    type RMSProp(model: Model, ?learningRate: Tensor, ?decay: double) =
        inherit ModelOptimizer(model)
        let learningRate = defaultArg learningRate (dsharp.tensor(1e-3))
        /// <summary>TBD</summary>
        override o.updateRule name t = failwith "TBD"

    type AdaDelta(model: Model) =
        inherit ModelOptimizer(model)
        /// <summary>TBD</summary>
        override o.updateRule name t = failwith "TBD"; failwith "TBD"

    type OptimizerWeightStepState() =
        member _.Item with get (parameter: Parameter) : Tensor = failwith "TBD"
        member _.weight : Tensor = failwith "TBD" 
        member _.grad : Tensor = failwith "TBD" 
        member _.step with get () : Tensor = failwith "TBD"  and set (v: Tensor) = failwith "TBD"

    type OptimizerState() =
        member _.Item 
            with get (state: OptimizerWeightStepState, parameter: Parameter) : Tensor = failwith "TBD"
            and set (state: OptimizerWeightStepState, parameter: Parameter) (v: Tensor) = failwith "TBD"

    type ParameterGroupOptimizer() =
        inherit Optimizer()
        /// <summary>TBD</summary>
        override o.updateRule name t = failwith "TBD"

    type ParameterGroupOptimizerBuilder() =
        member _.makeParameter(name: string, initial: Tensor) : Parameter = failwith "TBD"
        member _.makeStateParameter(name: string) : Parameter = failwith "TBD"
        member _.appendCallback(callback: OptimizerWeightStepState * OptimizerState -> unit) = failwith "TBD"
        member _.makeOptimizer() = failwith "TBD"

    type Embedding(embeddings: Tensor) =
        member val t = 
            embeddings.unstack() 
            |> Array.map (fun ab -> match ab.unstack() with [| a; b |] -> KeyValuePair(a,b) | _ -> failwith "expected pair")
            |> Dictionary
            with get, set
        
        new (vocabularySize: int, embeddingSize: int, embeddingsInitializer: Shape -> Tensor) =
            Embedding(failwith "TBD")

        new (vocabularySize: int, embeddingSize: int) =
            Embedding(failwith "TBD")

        member e.Item with get (v: Tensor) = e.t.[v]
        member e.embeddings 
             with get() = embeddings
             and set(embeddings: Tensor) = 
                 e.t <- Embedding(embeddings).t

    let truncatedNormalInitializer(standardDeviation: Tensor) : Shape -> Tensor = failwith "TBD"

    type TangentVector(x:obj) = class end

