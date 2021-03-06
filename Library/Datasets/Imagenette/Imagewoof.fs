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

namespace Datasets
// Original source:
// "Imagenette"
// Jeremy Howard
// https://github.com/fastai/Imagenette

(*


open DiffSharp

type Imagewoof {
  /// Type of the collection of non-collated batches.
  type Batches = Slices<Sampling<[(file: Uri, label: int32)], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  type Training = LazyMapSequence<
    TrainingEpochs<[(file: Uri, label: int32)], Entropy>,
    LazyMapSequence<Batches, LabeledImage>
  >
  /// The type of the validation data, represented as a collection of batches.
  type Validation = LazyMapSequence<Slices<[(file: Uri, label: int32)]>, LabeledImage>
  /// The training epochs.
  let training: Training
  /// The validation batches.
  let validation: Validation

  /// Creates an instance with `batchSize`.
  ///
  /// - Parameters:
  ///   - batchSize= Number of images provided per batch.
  ///   - entropy: A source of randomness used to shuffle sample
  ///     ordering.  It  will be stored in `self`, so if it is only pseudorandom
  ///     and has value semantics, the sequence of epochs is deterministic and not
  ///     dependent on other operations.
  ///   - device= The Device on which resulting Tensors from this dataset will be placed, as well
  ///     as where the latter stages of any conversion calculations will be performed.
  public init(batchSize: int, entropy: Entropy, device: Device) = 
    self.init(
      batchSize=batchSize, entropy: entropy, device=device, inputSize= ImagenetteSize.resized320,
      outputSize=224)


  /// Creates an instance with `batchSize` on `device` using `remoteBinaryArchiveLocation`.
  ///
  /// - Parameters:
  ///   - batchSize= Number of images provided per batch.
  ///   - entropy: A source of randomness used to shuffle sample ordering.  It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is deterministic and not dependent
  ///     on other operations.
  ///   - device= The Device on which resulting Tensors from this dataset will be placed, as well
  ///     as where the latter stages of any conversion calculations will be performed.
  ///   - inputSize= Which Imagenette image size variant to use.
  ///   - outputSize=The square width and height of the images returned from this dataset.
  ///   - localStorageDirectory: Where to place the downloaded and unarchived dataset.
  public init(
    batchSize: int, entropy: Entropy, device: Device, inputSize= ImagenetteSize,
    outputSize=Int,
    localStorageDirectory: Uri = DatasetUtilities.defaultDirectory
       </> ("Imagewoof")
  ) = 
    try
      // Training data
      let trainingSamples = try loadImagenetteTrainingDirectory(
        inputSize= inputSize, localStorageDirectory=localStorageDirectory, base: "imagewoof")

      let mean = Tensor([0.485, 0.456, 0.406], device=device)
      let standardDeviation = Tensor([0.229, 0.224, 0.225], device=device)

      training = TrainingEpochs(samples: trainingSamples, batchSize=batchSize, entropy: entropy)
         |> Seq.map (fun batches -> LazyMapSequence<Batches, LabeledImage> in
          batches |> Seq.map {
            makeImagenetteBatch(
              samples: $0, outFeatures=outputSize, mean: mean, stddev=standardDeviation,
              device=device)



      // Validation data
      let validationSamples = try loadImagenetteValidationDirectory(
        inputSize= inputSize, localStorageDirectory=localStorageDirectory, base: "imagewoof")

      validation = validationSamples.inBatches(of: batchSize) |> Seq.map {
        makeImagenetteBatch(
          samples: $0, outFeatures=outputSize, mean: mean, stddev=standardDeviation,
          device=device)

    with
      fatalError($"Could not load Imagewoof dataset: {error}")




extension Imagewoof: ImageClassificationData where Entropy = SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`, using the SystemRandomNumberGenerator.
  public init(batchSize: int, on device: Device = Device.default) = 
    self.init(batchSize=batchSize, device=device)


  /// Creates an instance with `batchSize`, `inputSize`, and `outputSize`, using the
  /// SystemRandomNumberGenerator.
  public init(
    batchSize: int, inputSize= ImagenetteSize, outFeatures=Int, on device: Device = Device.default
  ) = 
    self.init(
      batchSize=batchSize, device=device,
      inputSize= inputSize, outFeatures=outputSize)


*)
