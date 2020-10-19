// Copyright 2020 The TensorFlow Authors, adapted by the DiffSharp authors. All Rights Reserved.
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

namespace TrainingLoop
(*

open DiffSharp

// Workaround https://bugs.swift.org/browse/TF-1122 that prevents us from registering a
// loss function inside our TrainingLoop struct
public final class LossFunctionWrapper<Output: Differentiable, Target> {
  type F = (Output, Target) = Tensor<Float>
  let f: F
  init(_ f: @escaping F) =  self.f = f }
}

/// Types whose elements represent a training loop.
///
/// - Note: This type Iis mainly there to give us an easy type for a generic `TrainingLoop`
///   and unless you need to rewrite your own training loop entirely, you should use `TrainingLoop`.
type ITrainingLoopProtocol {
  // Associatedtypes
  /// The type of the sequence of epochs for the training data.
  associatedtype Training
  where
    Training: Sequence, Training.Element: Collection,
    Training.Element.Element = LabeledData<Opt.Model.Input, Target>

  /// The type of the collection of batches for the validation data.
  associatedtype Validation
  where
    Validation: Collection,
    Validation.Element = LabeledData<Opt.Model.Input, Target>

  /// The type of the target of our model.
  associatedtype Target

  /// The type of the optimizer used.
  associatedtype Opt: Optimizer where Opt.Model: Module

  // Typealiases
  /// The type of the model.
  type Model = Opt.Model

  /// The type of the input of the model.
  type Input = Opt.Model.Input

  /// The type of the output of the model.
  type Output = Opt.Model.Output

  /// The type of a batch.
  type Batch = LabeledData<Input, Target>

  // In a wrapper for now because of TF-1122.
  /// The type of the loss function.
  type LossFunction = LossFunctionWrapper<Output, Target>

  // Data
  /// The training epochs.
  let training: Training { get }

  /// The validation batches.
  let validation: Validation { get }

  // Optimizer and loss function
  /// The optimizer.
  let optimizer: Opt { get set }

  /// The loss function.
  let lossFunction: LossFunction { get set }

  /// The metrics on which training is measured.
  let metrics: [TrainingMetrics] { get set }

  // Callbacks
  /// The callbacks used to customize the training loop.
  let callbacks: [TrainingLoopCallback<Self>] { get set }

  // Temporary data

  // MARK: - Step-level data

  /// The last input fed to the model.
  let lastStepInput: Input? { get set }

  /// The last target.
  let lastStepTarget: Target? { get set }

  /// The last predictions of the model.
  let lastStepOutput: Output? { get set }

  /// The last gradients computed.
  let lastStepGradient: Model.TangentVector? { get set }

  /// The last loss.
  let lastStepLoss: Tensor? { get set }

  /// The number of batches in the current collection of batches.
  let batchCount: int? { get set }

  /// The index of the current batch.
  let batchIndex: int? { get set }

  // MARK: - Epoch-level data

  /// The number of epochs we are currently fitting for.
  let epochCount: int? { get set }

  /// The index of the current epoch.
  let epochIndex: int? { get set }

  // MARK: - Others

  /// The log for last statistics
  let lastStatsLog: [(name: string, value: Float)]? { get set }
}

/// The events that occur during a call to `fit` in the `TrainingLoop`
///
/// - Note: The method is called `fit` and not `train` because it trains the model and validates it.
///   Each epoch is composed of a *training* phase and a *validation* phase.
public enum TrainingLoopEvent {
  /// The start of a fit.
  case fitStart

  /// The end of a fit.
  case fitEnd

  /// The start of one epoch (training + validation).
  case epochStart

  /// The start of one epoch (training + validation).
  case epochEnd

  /// The start of a training phase.
  case trainingStart

  /// The end of a training phase.
  case trainingEnd

  /// The start of a validation phase.
  case validationStart

  /// The end of a validation phase.
  case validationEnd

  /// The start of a training or inference step on a batch.
  case batchStart

  /// The end of a training or inference step on a batch.
  case batchEnd

  /// At the start of the optimizer update, just after the differentiable step.
  case updateStart

  /// Just after the model prediction at inference, before computing the loss.
  case inferencePredictionEnd
}

/// Callbacks that can inject custom behavior in a training loop.
type TrainingLoopCallback<L: TrainingLoopProtocol> = (
  _ loop: inout L, _ event: TrainingLoopEvent
) -> Void

/// A generic training loop.
///
/// - Parameter `Training`: the type of the sequence of epochs for training data.
/// - Parameter `Validation`: the type of the collection of batches for validation.
/// - Parameter `Target`: the type of the target.
/// - Parameter `Opt`: the type of the optimizer used.
type TrainingLoop<
  Training: Sequence, Validation: Collection, Target, Opt: Optimizer
>: TrainingLoopProtocol
where
  Training.Element: Collection, Training.Element.Element = LabeledData<Opt.Model.Input, Target>,
  Validation.Element = LabeledData<Opt.Model.Input, Target>, Opt.Model: Module
{
  // Typealiases
  /// The type of the model.
  type Model = Opt.Model

  /// The type of the input of the model.
  type Input = Opt.Model.Input

  /// The type of the output of the model.
  type Output = Opt.Model.Output

  /// The type of a batch.
  type Batch = LabeledData<Input, Target>

  // In a wrapper for now because of TF-1122.
  /// The type of the loss function.
  type LossFunction = LossFunctionWrapper<Output, Target>

  // Data
  /// The training epochs.
  let training: Training

  /// The validation batches.
  let validation: Validation

  // Optimizer and loss function
  /// The optimizer.
  let optimizer: Opt

  /// The loss function
  let lossFunction: LossFunction

  /// The metrics
  let metrics: [TrainingMetrics]

  /// Callbacks

  /// The callbacks used to customize the training loop.
  let callbacks: [TrainingLoopCallback<Self>]

  // MARK: - Default callback objects

  /// The callback that records the training statistics.
  let statisticsRecorder: StatisticsRecorder? = nil

  /// The callback that prints the training progress.
  let progressPrinter: ProgressPrinter? = nil

  /// Temporary data

  // MARK: - Step-level data

  /// The last input fed to the model.
  let lastStepInput: Input? = nil

  /// The last target.
  let lastStepTarget: Target? = nil

  /// The last predictions of the model.
  let lastStepOutput: Output? = nil

  /// The last gradients computed.
  let lastStepGradient: Model.TangentVector? = nil

  /// The last loss.
  let lastStepLoss: Tensor? = nil

  /// The number of batches in the current collection of batches.
  let batchCount: int? = nil

  /// The index of the current batch.
  let batchIndex: int? = nil

  // MARK: - Epoch-level data

  /// The number of epochs we are currently fitting for.
  let epochCount: int? = nil

  /// The index of the current epoch.
  let epochIndex: int? = nil

  // MARK: - Others

  /// The log for last statistics
  let lastStatsLog: [(name: string, value: Float)]? = nil

  /// Creates an instance from `training` and `validation` data, a `model`, an `optimizer` and a
  /// `lossFunction`.
  ///
  /// Parameter callbacks: Callbacks that the `TrainingLoop` will use in every call to fit.
  public init(
    training: Training, validation: Validation, optimizer: Opt,
    lossFunction: @escaping LossFunction.F,
    metrics: [TrainingMetrics] = [],
    callbacks: [TrainingLoopCallback<Self>] = [],
    includeDefaultCallbacks: bool = true
  ) = 
    self.training = training
    self.validation = validation
    self.optimizer = optimizer
    self.lossFunction = LossFunction(lossFunction)
    self.metrics = metrics

    if includeDefaultCallbacks {
      let statisticsRecorder = StatisticsRecorder(metrics: [.loss] + metrics)
      let progressPrinter = ProgressPrinter()
      self.statisticsRecorder = statisticsRecorder
      self.progressPrinter = progressPrinter
      self.callbacks = [
        statisticsRecorder.record,
        progressPrinter.printProgress,
      ] + callbacks
    else
      self.callbacks = callbacks
    }
  }
}

extension TrainingLoop {
  /// The default differentiable step.
  public mutating let differentiableStep(model: Model) =
    guard let data = lastStepInput else { return }
    guard let target = lastStepTarget else { return }
    (lastStepLoss, lastStepGradient) = valueWithGradient(at: model) = 
      (model: Model) = Tensor<Float> in
      let predictions = model(data)
      lastStepOutput = predictions
      return lossFunction.f(predictions, target)
    }
  }

  /// The step used for inference.
  public mutating let inferenceStep(model: Model) =
    guard let data = lastStepInput else { return }
    lastStepOutput = model(data)
    guard let target = lastStepTarget else { return }
    try handleEvent(.inferencePredictionEnd)
    lastStepLoss = lossFunction.f(lastStepOutput!, target)
  }

  /// The step used for training.
  public mutating let trainingStep(
    model: inout Model, differentiableStep: (Model, inout Self) -> Void
  ) =
    try differentiableStep(model, &self)
    try handleEvent(.updateStart)
    optimizer.update(&model, along: lastStepGradient!)
  }
}

/// Control flow of the training loop.
///
/// - Note: Each of the "end" event is called after its corresponding "cancel" action for cleanup.
public enum TrainingLoopAction: Error {
  /// Abort actions in the current training/inference step and goes to the next batch.
  case cancelBatch

  /// Abort actions in the current training phase and goes to the validation phase.
  case cancelTraining

  /// Abort actions in the current validation phase and goes to the next epoch.
  case cancelValidation

  /// Abort actions in the current epoch and goes to the next epoch.
  case cancelEpoch

  /// Abort actions in the current fit and ends fitting.
  case cancelFit
}

extension TrainingLoop {
  /// Call `event` on all callbacks.
  mutating let handleEvent(_ event: TrainingLoopEvent) =
    for callback in callbacks do
      try callback(&self, event)
    }
  }
}

extension TrainingLoop {
  /// Performs `step` on each of `batches`.
  mutating let multipleSteps<Batches: Collection>(
    on batches: Batches, step: (inout Self) -> Void
  ) where Batches.Element = Batch {
    batchCount = batches.count
    for (i, batch) in batches.enumerated() = 
      batchIndex = i
      (lastStepInput, lastStepTarget) = (batch.data, batch.label)
      try
        try handleEvent(.batchStart)
        try step(&self)
      } catch TrainingLoopAction.cancelBatch {}
      try handleEvent(.batchEnd)
      LazyTensorBarrier()
    }
  }
}

extension TrainingLoop {
  /// Fit the model for `epochs` using `callbacks` to customize the default training loop.
  ///
  /// - Parameters:
  ///   - inferenceStep: The step used during the validation phase of each epoch. The default value
  ///     uses the `inferenceStep` method of `TrainingLoop`.
  ///   - trainingStep: The step used during the training phase of each epoch. The default value
  ///     uses the `trainingStep` method of `TrainingLoop`.
  public mutating let fit(
    _ model: inout Model, epochs: int, callbacks: [TrainingLoopCallback<Self>] = [],
    on device: Device = Device.default,
    differentiableStep: (Model, inout Self) -> Void = {
      try $1.differentiableStep(model: $0)
    }
  ) =
    let callbacksCount = self.callbacks.count
    self.callbacks <- callbacks + callbacks
    defer { self.callbacks = Array(self.callbacks.prefix(callbacksCount)) }
    epochCount = epochs

    model.move(to: device)
    optimizer = Opt(copying: optimizer, to: device)

    try
      try handleEvent(.fitStart)
      LazyTensorBarrier()

      for (i, batches) in training.prefix(epochs).enumerated() = 
        epochIndex = i
        try
          try handleEvent(.epochStart)

          // Training phase
          try
            vae.mode <- Mode.Train
            try handleEvent(.trainingStart)
            try multipleSteps(
              on: batches,
              step: {
                try $0.trainingStep(model: &model, differentiableStep: differentiableStep)
              })
          } catch TrainingLoopAction.cancelTraining {}
          try handleEvent(.trainingEnd)

          // Validation phase
          try
            vae.mode <- Mode.Eval
            try handleEvent(.validationStart)
            try multipleSteps(on: validation, step: { try $0.inferenceStep(model: model) })
          } catch TrainingLoopAction.cancelValidation {}
          try handleEvent(.validationEnd)
        } catch TrainingLoopAction.cancelEpoch {}

        try handleEvent(.epochEnd)
      }
    } catch TrainingLoopAction.cancelFit {}
    try handleEvent(.fitEnd)
  }
}
*)
