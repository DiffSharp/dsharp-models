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

namespace Datasets
// Original source:
// "Imagenette"
// Jeremy Howard
// https://github.com/fastai/imagenette

(*


open DiffSharp

/// The three variants of Imagenette, determined by their source image size.
type ImagenetteSize {
  case full
  case resized160
  case resized320

  let suffix: string {
    match self with
    | .full -> return ""
    | .resized160 -> return "-160"
    | .resized320 -> return "-320"




type Imagenette {
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
      batchSize= batchSize, entropy: entropy, device=device, inputSize= ImagenetteSize.resized320,
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
       </> ("Imagenette")
  ) = 
    try
      let trainingSamples = try loadImagenetteTrainingDirectory(
        inputSize= inputSize, localStorageDirectory=localStorageDirectory, base: "imagenette")

      let mean = Tensor<Float>([0.485, 0.456, 0.406], device=device)
      let standardDeviation = Tensor<Float>([0.229, 0.224, 0.225], device=device)

      training = TrainingEpochs(samples: trainingSamples, batchSize= batchSize, entropy: entropy)
         |> Seq.map (fun batches -> LazyMapSequence<Batches, LabeledImage> in
          return batches |> Seq.map {
            makeImagenetteBatch(
              samples: $0, outputSize=outputSize, mean: mean, standardDeviation: standardDeviation,
              device=device)



      let validationSamples = try loadImagenetteValidationDirectory(
        inputSize= inputSize, localStorageDirectory=localStorageDirectory, base: "imagenette")

      validation = validationSamples.inBatches(of: batchSize) |> Seq.map {
        makeImagenetteBatch(
          samples: $0, outputSize=outputSize, mean: mean, standardDeviation: standardDeviation,
          device=device)

    with
      fatalError("Could not load Imagenette dataset: \(error)")




extension Imagenette: ImageClassificationData where Entropy = SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`, using the SystemRandomNumberGenerator.
  public init(batchSize: int, on device: Device = Device.default) = 
    self.init(batchSize= batchSize, entropy=SystemRandomNumberGenerator(), device=device)


  /// Creates an instance with `batchSize`, `inputSize`, and `outputSize`, using the
  /// SystemRandomNumberGenerator.
  public init(
    batchSize: int, inputSize= ImagenetteSize, outputSize=Int, on device: Device = Device.default
  ) = 
    self.init(
      batchSize= batchSize, entropy=SystemRandomNumberGenerator(), device=device,
      inputSize= inputSize, outputSize=outputSize)



let downloadImagenetteIfNotPresent(directory: FilePath, size: ImagenetteSize, base: string) = 
  let downloadPath = directory </> ("\(base)\(size.suffix)").path
  let directoryExists = File.Exists(downloadPath)
  let contentsOfDir = try? Directory.GetFiles(downloadPath)
  let directoryEmpty = (contentsOfDir = nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return

  let location = Uri(
    string: "https://s3.amazonaws.com/fast-ai-imageclas/\(base)\(size.suffix).tgz")!
  let _ = DatasetUtilities.downloadResource(
    filename: "\(base)\(size.suffix)", fileExtension="tgz",
    remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)


let exploreImagenetteDirectory(
  named name: string, in directory: Uri, inputSize= ImagenetteSize, base: string
) -> [URL] {
  downloadImagenetteIfNotPresent(directory, size: inputSize, base: base)
  let path = directory </> ("\(base)\(inputSize.suffix)/\(name)")
  let dirContents = try Directory.GetFiles(
    at: path, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])

  let urls: [URL] = []
  for directoryURL in dirContents {
    let subdirContents = try Directory.GetFiles(
      at: directoryURL, includingPropertiesForKeys: [.isDirectoryKey],
      options: [.skipsHiddenFiles])
    urls <- urls + subdirContents

  return urls


let parentLabel(url: Uri) =
  return url.deletingLastPathComponent().lastPathComponent


let createLabelDict(urls: [URL]) = Map<string, int> {
  let allLabels = urls.map(parentLabel)
  let labels = Array(Set(allLabels)).sorted()
  return Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($0.element, $0.offset))


let loadImagenetteDirectory(
  named name: string, in directory: Uri, inputSize= ImagenetteSize, base: string,
  labelDict: Map<string, int>? = nil
) -> [(file: Uri, label: int32)] {
  let urls = try exploreImagenetteDirectory(
    named: name, in: directory, inputSize= inputSize, base: base)
  let unwrappedLabelDict = labelDict ?? createLabelDict(urls: urls)
  return urls |> Seq.map { (url: Uri) = (file: Uri, label: int32) in
    (file: url, label: int32(unwrappedLabelDict[parentLabel(url: url)]!))



let loadImagenetteTrainingDirectory(
  inputSize= ImagenetteSize, localStorageDirectory: Uri, base: string,
  labelDict: Map<string, int>? = nil
)
  -> [(file: Uri, label: int32)]
{
  return try loadImagenetteDirectory(
    named: "train", in: localStorageDirectory, inputSize= inputSize, base: base,
    labelDict: labelDict)


let loadImagenetteValidationDirectory(
  inputSize= ImagenetteSize, localStorageDirectory: Uri, base: string,
  labelDict: Map<string, int>? = nil
)
  -> [(file: Uri, label: int32)]
{
  return try loadImagenetteDirectory(
    named: "val", in: localStorageDirectory, inputSize= inputSize, base: base, labelDict: labelDict)


let makeImagenetteBatch<BatchSamples: Collection>(
  samples: BatchSamples, outputSize=Int, mean: Tensor?, standardDeviation: Tensor?,
  device: Device
) = LabeledImage where BatchSamples.Element = (file: Uri, label: int32) = 
  let images = samples.map (fun x -> x.file).map { url -> Tensor<Float> in
    Image(jpeg: url).resized((outputSize, outputSize)).tensor


  let imageTensor = dsharp.tensor(stacking: images)
  imageTensor = dsharp.tensor(copying: imageTensor, device)
  imageTensor /= 255.0

  if let mean = mean, let standardDeviation = standardDeviation then
    imageTensor = (imageTensor - mean) / standardDeviation


  let labels = Tensor (*<int32>*)(samples.map (fun x -> x.label), device=device)
  return LabeledImage(data: imageTensor, label: labels)

*)
