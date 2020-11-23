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

module Benchmark.LayerSuite

open BenchmarkDotNet.Attributes
open BenchmarkDotNet.Configs
open DiffSharp
open DiffSharp.Model
open Models.ImageClassification.DenseNet121
open Models.ImageClassification.EfficientNet
open Models.ImageClassification.LeNet_5

let mnistInput = [28; 28; 1]
let mnistOutput = [10]

let cifarInput = [32; 32; 3]
let cifarOutput = [10]

let imageNetInput = [224; 224; 3]
let imageNetOutput = [1000]

let makeRandomTensor(batchSize: int, dimensions: seq<int>, device: Device) =
    let allDimensions = [| yield batchSize; yield! dimensions |]
    dsharp.seed(seed=0xffeffe)
    let tensor = dsharp.randn(allDimensions, mean=0.5, stddev=0.1, device=device)
    tensor

let makeEvalBenchmark (layer: Model, inputDimensions: seq<int>, outputDimensions: seq<int>) = fun (device:Device, batchSize) ->
    layer.move(device=device)
    let input = makeRandomTensor(batchSize, inputDimensions, device)
    let mutable sink = makeRandomTensor(batchSize, outputDimensions, device)
    let result = layer.forward(input)
    // Force materialization of the lazy results.
    sink <- sink + result

let makeGradientBenchmark(layer: Model, inputDimensions: seq<int>, outputDimensions: seq<int>) = fun (device:Device, batchSize) ->
    layer.move(device=device)

    let input = makeRandomTensor(batchSize, inputDimensions, device)
    let output = makeRandomTensor(batchSize, outputDimensions, device)
    let mutable sink = dsharp.zero(device=device)
    let result = layer.grad(input, (fun t -> dsharp.mseLoss(t, output)))
    sink <- sink + result


[<ShortRunJob>]
[<MarkdownExporterAttribute.GitHub; AsciiDocExporter; HtmlExporter; CsvExporter; RPlotExporter>]
[<GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)>]
[<CategoriesColumn; BaselineColumn>]
type LayerBenchmarks() =

    [<Benchmark>]
    member _.DenseNet121_Eval() =
        makeEvalBenchmark(DenseNet121(classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.DenseNet121_Gradient() =
        makeGradientBenchmark(DenseNet121(classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB0_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB0, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB0_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB0, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB1_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB1, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB1_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB1, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB2_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB2, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB2_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB2, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB3_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB3, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB3_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB3, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB4_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB4, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB4_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB4, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB5_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB5, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB5_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB5, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB6_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB6, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB6_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB6, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB7_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB7, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB7_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB7, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB8_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetB8, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetB8_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetB8, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetL2_Eval() =
        makeEvalBenchmark(EfficientNet.Create(kind=EfficientnetL2, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.EfficientNetL2_Gradient() =
        makeGradientBenchmark(EfficientNet.Create(kind=EfficientnetL2, classCount=1000), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.LeNet_Eval() =
        makeEvalBenchmark(LeNet(), imageNetInput, imageNetOutput) (Device.CPU, 4)

    [<Benchmark>]
    member _.LeNet_Gradient() =
        makeGradientBenchmark(LeNet(), mnistInput, mnistOutput) (Device.CPU, 4)
(*
open Models.ImageClassification

let MobileNetV1Suites = [
  makeLayerSuite(
    name="MobileNetV1",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    MobileNetV1(classCount=10)
  },
  makeLayerSuite(
    name="MobileNetV1",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    MobileNetV1(classCount=1000)
  },
]
*)
(*
open Models.ImageClassification

let MobileNetV2Suites = [
  makeLayerSuite(
    name="MobileNetV2",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    MobileNetV2(classCount=10)
  },
  makeLayerSuite(
    name="MobileNetV2",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    MobileNetV2(classCount=1000)
  },
]
*)
(*
open Models.ImageClassification

let MobileNetV3Suites = [
  makeLayerSuite(
    name="MobileNetV3Small",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    MobileNetV3Small(classCount=10)
  },
  makeLayerSuite(
    name="MobileNetV3Small",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    MobileNetV3Small(classCount=1000)
  },
  makeLayerSuite(
    name="MobileNetV3Large",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    MobileNetV3Large(classCount=10)
  },
  makeLayerSuite(
    name="MobileNetV3Large",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    MobileNetV3Large(classCount=1000)
  },
]
*)
(*
open Models.ImageClassification

let ResNetSuites = [
  //
  // Cifar input dimensions. 
  //
  makeLayerSuite(
    name="ResNet18",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    ResNet(classCount=10, depth: ResNet18, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name="ResNet34",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    ResNet(classCount=10, depth: ResNet34, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name="ResNet50",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    ResNet(classCount=10, depth: ResNet50, downsamplingInFirstStage: true, useLaterStride: false)
  },
  makeLayerSuite(
    name="ResNet56",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    ResNet(classCount=10, depth: ResNet56, downsamplingInFirstStage: true, useLaterStride: false)
  },
  makeLayerSuite(
    name="ResNet101",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    ResNet(classCount=10, depth: ResNet101, downsamplingInFirstStage: true)
  },
  makeLayerSuite(
    name="ResNet152",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    ResNet(classCount=10, depth: ResNet152, downsamplingInFirstStage: true)
  },
  //
  // ImageNet dimensions.
  //
  makeLayerSuite(
    name="ResNet18",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNet(classCount=1000, depth: ResNet18)
  },
  makeLayerSuite(
    name="ResNet34",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNet(classCount=1000, depth: ResNet34)
  },
  makeLayerSuite(
    name="ResNet50",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNet(classCount=1000, depth: ResNet50, useLaterStride: false)
  },
  makeLayerSuite(
    name="ResNet56",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNet(classCount=1000, depth: ResNet56)
  },
  makeLayerSuite(
    name="ResNet101",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNet(classCount=1000, depth: ResNet101)
  },
  makeLayerSuite(
    name="ResNet152",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNet(classCount=1000, depth: ResNet152)
  },
]
*)
(*
open Models.ImageClassification

let ResNetV2Suites = [
  makeLayerSuite(
    name="ResNet18v2",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNetV2(classCount=1000, depth: ResNet18)
  },
  makeLayerSuite(
    name="ResNet34v2",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNetV2(classCount=1000, depth: ResNet34)
  },
  makeLayerSuite(
    name="ResNet50v2",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNetV2(classCount=1000, depth: ResNet50)
  },
  makeLayerSuite(
    name="ResNet101v2",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNetV2(classCount=1000, depth: ResNet101)
  },
  makeLayerSuite(
    name="ResNet152v2",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ResNetV2(classCount=1000, depth: ResNet152)
  },
]
*)
(*
open Models.ImageClassification

let ShuffleNetV2Suites = [
  makeLayerSuite(
    name="ShuffleNetV2x05",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ShuffleNetV2.Create(kind=ShuffleNetV2x05)
  },
  makeLayerSuite(
    name="ShuffleNetV2x10",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ShuffleNetV2.Create(kind=ShuffleNetV2x10)
  },
  makeLayerSuite(
    name="ShuffleNetV2x15",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ShuffleNetV2.Create(kind=ShuffleNetV2x15)
  },
  makeLayerSuite(
    name="ShuffleNetV2x20",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    ShuffleNetV2.Create(kind=ShuffleNetV2x20)
  },
]
*)
(*
open Models.ImageClassification

let SqueezeNetSuites = [
  makeLayerSuite(
    name="SqueezeNetV1_0",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    SqueezeNetV1_0(classCount=1000)
  },
  makeLayerSuite(
    name="SqueezeNetV1_1",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    SqueezeNetV1_1(classCount=1000)
  },
]
*)
(*
open Models.ImageClassification

let VGGSuites = [
  makeLayerSuite(
    name="VGG16",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    VGG16(classCount=1000)
  },
  makeLayerSuite(
    name="VGG19",
    inputDimensions=imageNetInput,
    outputDimensions=imageNetOutput
  ) = 
    VGG19(classCount=1000)
  },
]
*)
(*
open Models.ImageClassification

let WideResNetSuites = [
  makeLayerSuite(
    name="WideResNet16",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet16)
  },
  makeLayerSuite(
    name="WideResNet16k10",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet16k10)
  },
  makeLayerSuite(
    name="WideResNet22",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet22)
  },
  makeLayerSuite(
    name="WideResNet22k10",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet22k10)
  },
  makeLayerSuite(
    name="WideResNet28",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet28)
  },
  makeLayerSuite(
    name="WideResNet28k12",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet28k12)
  },
  makeLayerSuite(
    name="WideResNet40k1",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet40k1)
  },
  makeLayerSuite(
    name="WideResNet40k2",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet40k2)
  },
  makeLayerSuite(
    name="WideResNet40k4",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet40k4)
  },
  makeLayerSuite(
    name="WideResNet40k8",
    inputDimensions=cifarInput,
    outputDimensions=cifarOutput
  ) = 
    WideResNet.Create(kind= WideResNet40k8)
  },
]
*)
