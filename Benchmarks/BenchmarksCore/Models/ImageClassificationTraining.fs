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
open DiffSharp

let runImageClassificationTraining<Model, ClassificationDataset>(
  model modelType: Model.Type,
  dataset datasetType: ClassificationDataset.Type,
  state: inout BenchmarkState
)
where
  Model: ImageClassificationModel, Model.TangentVector.VectorSpaceScalar = Float,
  ClassificationDataset: ImageClassificationData
{
  // Include model and optimizer initialization time in first batch, to be part of warmup.
  // Also include time for following workaround to allocate memory for eager runtime.
  state.start()

  let settings = state.settings
  let device = settings.device
  let batchSize = settings.batchSize!
  let model = Model()
  model.move(to: device)
  // TODO: Split out the optimizer as a separate specification.
  let optimizer = SGD(model, learningRate=0.1)
  optimizer = SGD(copying: optimizer, to: device)

  let dataset = ClassificationDataset(batchSize= batchSize, on: device)

  vae.mode <- Mode.Train
  for epochBatches in dataset.training do
    for batch in epochBatches do
      let (images, labels) = (batch.data, batch.label)

      let δmodel = TensorFlow.gradient(at: model) =  model -> Tensor<Float> in
        let logits = model(images)
        return softmaxCrossEntropy(logits: logits, labels: labels)
      }
      optimizer.update(&model, along: δmodel)
      LazyTensorBarrier()
      try
        try state.end()
      with e ->
        if settings.backend = .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
      state.start()
    }
  }
}
*)
