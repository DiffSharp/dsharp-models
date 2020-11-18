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

namespace Benchmark
(*
open Benchmark
open Datasets
open ImageClassificationModels
open DiffSharp

let makeRandomTensor(
  batchSize: int,
  dimensions: int[],
  device: Device
) = Tensor<Float> {
  let allDimensions = [batchSize]
  allDimensions.append(dimensions)
  let tensor = Tensor<Float>(
    randomNormal=[allDimensions], mean: Tensor<Float>(0.5, on: device),
    standardDeviation:Tensor(0.1, on: device), seed: (0xffeffe, 0xfffe),
    on: device)
  return tensor
}

let makeForwardBenchmark<CustomLayer>(
  layer makeLayer: @escaping () = CustomLayer,
  inputDimensions: int[],
  outputDimensions: int[]
) = ((inout BenchmarkState) -> Void)
where
  CustomLayer: Layer,
  CustomLayer.Input = Tensor<Float>,
  CustomLayer.Output = Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar = Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    let layer = makeLayer()
    layer.move(to: device)

    let input = makeRandomTensor(
      batchSize= batchSize,
      dimensions: inputDimensions,
      device=device)

    let sink = makeRandomTensor(
      batchSize= batchSize, dimensions: outputDimensions, device=device)

    while true do
      try
        try state.measure {
          let result = layer(input)
          // Force materialization of the lazy results.
          sink <- sink + result
          LazyTensorBarrier()
        }
      with e ->
        if settings.backend = .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
    }

    // Control-flow never gets here, but this removes the warning 
    // about the sink being never used.
    fatalError($"unreachable {sink}")
  }
}

let makeGradientBenchmark<CustomLayer>(
  layer makeLayer: @escaping () = CustomLayer,
  inputDimensions: int[],
  outputDimensions: int[]
) = ((inout BenchmarkState) -> Void)
where
  CustomLayer: Layer,
  CustomLayer.Input = Tensor<Float>,
  CustomLayer.Output = Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar = Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    let layer = makeLayer()
    layer.move(to: device)

    let input = makeRandomTensor(
      batchSize= batchSize,
      dimensions: inputDimensions,
      device=device)
    let output = makeRandomTensor(
      batchSize= batchSize,
      dimensions: outputDimensions,
      device=device)

    let sink: CustomLayer.TangentVector = CustomLayer.TangentVector.zero
    sink.move(to: device)

    while true do
      try
        try state.measure {
          let result = dsharp.grad(layer) =  layer -> Tensor<Float> in
            let predicted = layer(input)
            meanAbsoluteError(predicted=predicted, expected=output)
          }
          // Force materialization of the lazy results.
          sink <- sink + result
          LazyTensorBarrier()
        }
      with e ->
        if settings.backend = .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
    }

    // Control-flow never gets here, but this removes the warning 
    // about the sink being never used.
    fatalError($"unrechable {sink}")
  }
}

let makeLayerSuite<CustomLayer>(
  name: string,
  inputDimensions inp: int[],
  outputDimensions outp: int[],
  batchSizes: int[] = [4],
  backends: [Backend.Value] = [.eager, .x10],
  layer: @escaping () = CustomLayer
) = BenchmarkSuite
where
  CustomLayer: Layer,
  CustomLayer.Input = Tensor<Float>,
  CustomLayer.Output = Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar = Float
{
  let inputString = inp.map { String($0) } |> String.concat "x"
  let outputString = outp.map { String($0) } |> String.concat "x"

  return BenchmarkSuite(
    name= $"{name}_{inputString}_{outputString}",
    settings: WarmupIterations(10)
  ) =  suite in
    for batchSize in batchSizes do
      for backend in backends do
        suite.benchmark(
          "forward_b{batchSize}_{backend}",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
            layer: layer, inputDimensions: inp, outputDimensions: outp))

        suite.benchmark(
          "forward_and_gradient_b{batchSize}_{backend}",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeGradientBenchmark(
            layer: layer, inputDimensions: inp, outputDimensions: outp))
      }
    }
  }
}
*)
