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
open ArgumentParser
open Benchmark

type BenchmarkArguments: ParsableArguments {
  @OptionGroup()
  let arguments: Benchmark.BenchmarkArguments

  @Option(help: "Size of a single batch.")
  let batchSize: int?

  @Flag(help: "Use eager backend.")
  let eager: bool = false

  @Flag(help: "Use X10 backend.")
  let x10: bool = false

  @Flag(help: "Use CPU platform.")
  let cpu: bool = false

  @Flag(help: "Use GPU platform.")
  let gpu: bool = false

  @Flag(help: "Use TPU platform.")
  let tpu: bool = false

  @Flag(help: "Use synthetic data.")
  let synthetic: bool = false

  @Flag(help: "Use real data.")
  let real: bool = false

  @Option(help: "File path for dataset loading.")
  let datasetFilePath: string?

  public init() = }

  public init(
    arguments: Benchmark.BenchmarkArguments, batchSize: int?, eager: bool, x10: bool,
    cpu: bool, gpu: bool, tpu: bool, synthetic: bool, real: bool,
    datasetFilePath: string?
  ) = 
    self.arguments = arguments
    self.batchSize = batchSize
    self.eager = eager
    self.x10 = x10
    self.cpu = cpu
    self.gpu = gpu
    self.tpu = tpu
    self.synthetic = synthetic
    self.real = real
    self.datasetFilePath = datasetFilePath
  }

  public mutating let validate() =
    try arguments.validate()

    guard !(real && synthetic) else {
      throw ValidationError(
        "Can't specify both --real and --synthetic data sources.")
    }

    guard !(eager && x10) else {
      throw ValidationError(
        "Can't specify both --eager and --x10 backends.")
    }

    guard !(cpu && gpu) || !(cpu && tpu) || !(gpu && tpu) else {
      throw ValidationError(
        "Can't specify multiple platforms.")
    }
  }

  let settings: [BenchmarkSetting] {
    let settings = arguments.settings

    if let value = batchSize {
      settings.append(BatchSize(value))
    }
    if x10 {
      settings.append(Backend(.x10))
    }
    if eager {
      settings.append(Backend(.eager))
    }
    if cpu {
      settings.append(Platform(.cpu))
    }
    if gpu {
      settings.append(Platform(.gpu))
    }
    if tpu {
      settings.append(Platform(.tpu))
    }
    if synthetic {
      settings.append(Synthetic(true))
    }
    if real {
      settings.append(Synthetic(false))
    }
    if let value = datasetFilePath {
      settings.append(DatasetFilePath(value))
    }

    return settings
  }
}
*)
