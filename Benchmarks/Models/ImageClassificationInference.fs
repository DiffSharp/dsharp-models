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
open Models.ImageClassification
open DiffSharp

type IImageClassificationModel() = 
  inherit Model()
  static let preferredInputDimensions: int[] { get }
  static let outputLabels: int { get }

let runImageClassificationInference<Model, ClassificationDataset>(
  model modelType: Model.Type,
  dataset realDatasetType: ClassificationDataset.Type,
  state: inout BenchmarkState
)
where
  Model: ImageClassificationModel,
  ClassificationDataset: ImageClassificationData
{
  let settings = state.settings
  let device = settings.device
  let batchSize = settings.batchSize!
  let dataset = ClassificationDataset(batchSize=batchSize, device=device)
  let model = Model()
  model.move(device)

  for epochBatches in dataset.training do
    for batch in epochBatches do
      let images = batch.data

      try
        try state.measure {
          let _ = model(images)
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
  }
}
*)
