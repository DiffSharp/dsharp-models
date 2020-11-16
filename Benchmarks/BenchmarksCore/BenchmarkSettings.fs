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
open DiffSharp

type batchSize= BenchmarkSetting {
  let value: int
  init(value: int) = 
    self.value = value
  }
}

type Length: BenchmarkSetting {
  let value: int
  init(value: int) = 
    self.value = value
  }
}

type Synthetic: BenchmarkSetting {
  let value: bool
  init(value: bool) = 
    self.value = value
  }
}

type Backend: BenchmarkSetting {
  let value: Value
  init(value: Value) = 
    self.value = value
  }
  type Value {
    case x10
    case eager
  }
}

type Platform: BenchmarkSetting {
  let value: Value
  init(value: Value) = 
    self.value = value
  }
  type Value {
    case `default`
    case cpu
    case gpu
    case tpu
  }
}

type DatasetFilePath: BenchmarkSetting {
  let value: string
  init(value: string) = 
    self.value = value
  }
}

extension BenchmarkSettings {
  let batchSize: int? {
    return self[BatchSize.self]?.value
  }

  let length: int? {
    return self[Length.self]?.value
  }

  let synthetic: bool {
    if let value = self[Synthetic.self]?.value {
      return value
    else
      fatalError("Synthetic setting must have a default.")
    }
  }

  let backend: Backend.Value {
    if let value = self[Backend.self]?.value {
      return value
    else
      fatalError("Backend setting must have a default.")
    }
  }

  let platform: Platform.Value {
    if let value = self[Platform.self]?.value {
      return value
    else
      fatalError("Platform setting must have a default.")
    }
  }

  let device: Device {
    // Note: The line is needed, or all GPU memory
    // will be exhausted on initial allocation of the model.
    // TODO: Remove the following tensor workaround when above is fixed.
    let _ = _ExecutionContext.global

    match backend with
    | .eager ->
      match platform with
      | .default -> return Device.defaultTFEager
      | .cpu -> return Device(kind: .CPU, ordinal: 0, backend: .TF_EAGER)
      | .gpu -> return Device(kind: .GPU, ordinal: 0, backend: .TF_EAGER)
      | .tpu -> fatalError("TFEager is unsupported on TPU.")
      }
    | .x10 ->
      match platform with
      | .default -> return Device.defaultXLA
      | .cpu -> return Device(kind: .CPU, ordinal: 0, backend: .XLA)
      | .gpu -> return Device(kind: .GPU, ordinal: 0, backend: .XLA)
      | .tpu -> return (Device.allDevices.filter (fun x -> x..kind = .TPU }).first!
      }
    }
  }

  let datasetFilePath: string? {
    return self[DatasetFilePath.self]?.value
  }
}

let defaultSettings: BenchmarkSetting[] = [
  TimeUnit(.s),
  InverseTimeUnit(.s),
  Backend(.eager),
  Platform(.default),
  Synthetic(false),
  Columns([
    "name",
    "wall_time",
    "startup_time",
    "iterations",
    "avg_exp_per_second",
    "exp_per_second",
    "step_time_median",
    "step_time_min",
    "step_time_max",
  ]),
]
*)
