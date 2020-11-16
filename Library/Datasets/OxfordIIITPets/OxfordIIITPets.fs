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
// Original Source
// "The Oxford-IIIT Pet Dataset"
// Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman and C. V. Jawahar
// https://www.robots.ox.ac.uk/~vgg/data/pets/

(*

open DiffSharp

type OxfordIIITPets {
  /// Type of the collection of non-collated batches.
  type Batches = Slices<Sampling<[(file: Uri, annotation: Uri)], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  type Training = LazyMapSequence<
    TrainingEpochs<[(file: Uri, annotation: Uri)], Entropy>,
    LazyMapSequence<Batches, SegmentedImage>
  >
  /// The type of the validation data, represented as a collection of batches.
  type Validation = LazyMapSequence<
    Slices<[(file: Uri, annotation: Uri)]>, LabeledImage
  >
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
      batchSize= batchSize, entropy: entropy, device=device, imageSize: 224)


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
  ///   - imageSize: The square width and height of the images returned from this dataset.
  ///   - localStorageDirectory: Where to place the downloaded and unarchived dataset.
  public init(
    batchSize: int, entropy: Entropy, device: Device, imageSize: int,
    localStorageDirectory: Uri = DatasetUtilities.defaultDirectory
       </> ("OxfordIIITPets")
  ) = 
    try
      let trainingSamples = try loadOxfordIITPetsTraining(
        localStorageDirectory=localStorageDirectory)

      training = TrainingEpochs(samples: trainingSamples, batchSize= batchSize, entropy: entropy)
         |> Seq.map (fun batches -> LazyMapSequence<Batches, LabeledImage> in
          return batches |> Seq.map {
            makeBatch(samples: $0, imageSize: imageSize, device=device)



      let validationSamples = try loadOxfordIITPetsTraining(
        localStorageDirectory=localStorageDirectory)

      validation = validationSamples.inBatches(of: batchSize) |> Seq.map {
        makeBatch(samples: $0, imageSize: imageSize, device=device)

    with
      fatalError($"Could not load the Oxford IIIT Pets dataset: {error}")




extension OxfordIIITPets: ImageSegmentationData where Entropy = SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`, using the SystemRandomNumberGenerator.
  public init(batchSize: int, on device: Device = Device.default) = 
    self.init(batchSize= batchSize, device=device)


  /// Creates an instance with `batchSize`, `inputSize`, and `outputSize`, using the
  /// SystemRandomNumberGenerator.
  public init(batchSize: int, imageSize: int, on device: Device = Device.default) = 
    self.init(
      batchSize= batchSize, device=device,
      imageSize: imageSize)



let downloadOxfordIIITPetsIfNotPresent(directory: FilePath) = 
  let downloadPath = directory </> ("images").path
  let directoryExists = File.Exists(downloadPath)
  let contentsOfDir = try? Directory.GetFiles(downloadPath)
  let directoryEmpty = (contentsOfDir = nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return

  let remoteRoot = Uri("https://www.robots.ox.ac.uk/~vgg/data/pets/data/")!

  let _ = DatasetUtilities.downloadResource(
    filename: "images", fileExtension="tar.gz",
    remoteRoot=remoteRoot, localStorageDirectory: directory
  )

  let _ = DatasetUtilities.downloadResource(
    filename: "annotations", fileExtension="tar.gz",
    remoteRoot=remoteRoot, localStorageDirectory: directory
  )


let loadOxfordIIITPets(filename: string, in directory: Uri) -> [(
  file: Uri, annotation: Uri
)] {
  downloadOxfordIIITPetsIfNotPresent(directory)
  let imageURLs = getImageURLs(filename: filename, directory: directory)
  return imageURLs |> Seq.map { (imageURL: Uri) = (file: Uri, annotation: Uri) in
    (file: imageURL, annotation: makeAnnotationURL(imageURL: imageURL, directory: directory))



let makeAnnotationURL(imageURL: Uri, directory: Uri) = URL {
  let filename = imageURL.deletingPathExtension().lastPathComponent
  return directory </> ($"annotations/trimaps/{filename}.png")


let getImageURLs(filename: string, directory: Uri) = [URL] {
  let filePath = directory </> ($"annotations/{filename}")
  let imagesRootDirectory = directory </> ("images")
  let fileContents = try? String(contentsOf: filePath)
  let imageDetails = fileContents!.Split("\n")
  return imageDetails.map {
    let imagename = String($0[..<$0.firstIndex(of: " ")!])
    return imagesRootDirectory </> ($"{imagename}.jpg")



let loadOxfordIITPetsTraining(localStorageDirectory: Uri) -> [(file: Uri, annotation: Uri)]
{
  return try loadOxfordIIITPets(
    filename: "trainval.txt", in: localStorageDirectory)


let loadOxfordIIITPetsValidation(localStorageDirectory: Uri) -> [(
  file: Uri, annotation: Uri
)] {
  return try loadOxfordIIITPets(
    filename: "test.txt", in: localStorageDirectory)


let makeBatch<BatchSamples: Collection>(
  samples: BatchSamples, imageSize: int, device: Device
) = SegmentedImage where BatchSamples.Element = (file: Uri, annotation: Uri) = 
  let images = samples.map (fun x -> x.file).map { url -> Tensor<Float> in
    Image(jpeg: url).resized((imageSize, imageSize)).tensor[0..., 0..., 0..<3]


  let imageTensor = dsharp.tensor(stacking: images)
  imageTensor = dsharp.tensor(copying: imageTensor, device)
  imageTensor /= 255.0

  let annotations = samples.map (fun x -> x.annotation).map { url -> Tensor (*<int32>*) in
    Tensor (*<int32>*)(
      Image(jpeg: url).resized((imageSize, imageSize)).tensor[0..., 0..., 0...0] - 1)

  let annotationTensor = dsharp.tensor(stacking: annotations)
  annotationTensor = dsharp.tensor(copying: annotationTensor, device)

  return SegmentedImage(data: imageTensor, label: annotationTensor)

*)
