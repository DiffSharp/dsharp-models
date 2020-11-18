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
// "The CIFAR-10 dataset"
// Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
// https://www.cs.toronto.edu/~kriz/cifar.html

(*


open DiffSharp

type CIFAR10 {
  /// Type of the collection of non-collated batches.
  type Batches = Slices<Sampling<[(data: byte[], label: int32)], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  type Training = LazyMapSequence<
    TrainingEpochs<[(data: byte[], label: int32)], Entropy>,
    LazyMapSequence<Batches, LabeledImage>
  >
  /// The type of the validation data, represented as a collection of batches.
  type Validation = LazyMapSequence<Slices<[(data: byte[], label: int32)]>, LabeledImage>
  /// The training epochs.
  let training: Training
  /// The validation batches.
  let validation: Validation

  /// Creates an instance with `batchSize`.
  ///
  /// - Parameter entropy: a source of randomness used to shuffle sample 
  ///   ordering.  It  will be stored in `self`, so if it is only pseudorandom 
  ///   and has value semantics, the sequence of epochs is deterministic and not 
  ///   dependent on other operations.
  public init(batchSize: int, entropy: Entropy, device: Device) = 
    self.init(
      batchSize= batchSize,
      entropy: entropy,
      device=device,
      remoteBinaryArchiveLocation: Uri(
        string: "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/CIFAR10/cifar-10-binary.tar.gz")!, 
      normalizing=true)

  
  /// Creates an instance with `batchSize` on `device` using `remoteBinaryArchiveLocation`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering.  It  
  ///     will be stored in `self`, so if it is only pseudorandom and has value 
  ///     semantics, the sequence of epochs is deterministic and not dependent 
  ///     on other operations.
  ///   - normalizing: normalizes the batches with the mean and standard deviation
  ///     of the dataset iff `true`. Default value is `true`.
  public init(
    batchSize: int,
    entropy: Entropy,
    device: Device,
    remoteBinaryArchiveLocation: Uri, 
    localStorageDirectory: Uri = DatasetUtilities.defaultDirectory
       </> ("CIFAR10"), 
    normalizing: bool
  ){
    downloadCIFAR10IfNotPresent(remoteBinaryArchiveLocation, localStorageDirectory)
    
    let mean: Tensor?
    let standardDeviation=Tensor?
    if normalizing then
      mean = Tensor<Float>([0.4913996898, 0.4821584196, 0.4465309242], device=device)
      standardDeviation = Tensor<Float>([0.2470322324, 0.2434851280, 0.2615878417], device=device)

    
    // Training data
    let trainingSamples = loadCIFARTrainingFiles(in: localStorageDirectory)
    training = TrainingEpochs(samples: trainingSamples, batchSize= batchSize, entropy: entropy)
       |> Seq.map (fun batches -> LazyMapSequence<Batches, LabeledImage> in
        batches |> Seq.map{
          makeBatch(samples: $0, mean: mean, standardDeviation=standardDeviation, device=device)


      
    // Validation data
    let validationSamples = loadCIFARTestFile(in: localStorageDirectory)
    validation = validationSamples.inBatches(of: batchSize) |> Seq.map {
      makeBatch(samples: $0, mean: mean, standardDeviation=standardDeviation, device=device)




extension CIFAR10: ImageClassificationData where Entropy = SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`.
  public init(batchSize: int, on device: Device = Device.default) = 
    self.init(batchSize= batchSize, device=device)



let downloadCIFAR10IfNotPresent(from location: Uri, directory: FilePath) = 
  let downloadPath = directory </> ("cifar-10-batches-bin").path
  let directoryExists = File.Exists(downloadPath)
  let contentsOfDir = try? Directory.GetFiles(downloadPath)
  let directoryEmpty = (contentsOfDir = nil) || (contentsOfDir!.isEmpty)

  guard !directoryExists || directoryEmpty else { return

  let _ = DatasetUtilities.downloadResource(
    filename: "cifar-10-binary", fileExtension="tar.gz",
    remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)


let loadCIFARFile(named name: string, in directory: Uri) = [(data: byte[], label: int32)] {
  let path = directory </> ("cifar-10-batches-bin/{name}").path

  let imageCount = 10000
  guard let fileContents = 
      Data.ReadAllBytes(Uri(fileURLWithPath= path)) 
      with _ -> 
          printError("Could not read dataset file: {name}")
          exit(-1)

  guard fileContents.count = 30_730_000 else {
    printError(
      $"Dataset file {name} should have 30730000 bytes, instead had {fileContents.count}")
    exit(-1)


  let labeledImages: [(data: byte[], label: int32)] = []

  let imageByteSize = 3073
  for imageIndex in 0..imageCount-1 do
    let baseAddress = imageIndex * imageByteSize
    let label = int32(fileContents[baseAddress])
    let data = [byte](fileContents[(baseAddress + 1)..<(baseAddress + 3073)])
    labeledImages.append((data: data, label: label))


  return labeledImages


let loadCIFARTrainingFiles(in localStorageDirectory: Uri) = [(data: byte[], label: int32)] {
  let data = (1..<6).map =
    loadCIFARFile(named: "data_batch_\($0).bin", in: localStorageDirectory)

  return data.reduce([], +)


let loadCIFARTestFile(in localStorageDirectory: Uri) = [(data: byte[], label: int32)] {
  return loadCIFARFile(named: "test_batch.bin", in: localStorageDirectory)


let makeBatch<BatchSamples: Collection>(
  samples: BatchSamples, mean: Tensor?, standardDeviation=Tensor?, device: Device
) = LabeledImage where BatchSamples.Element = (data: byte[], label: int32) = 
  let bytes = samples |> Seq.map (fun x -> x.data).reduce(into: [], +=)
  let images = Tensor<byte>(shape=[samples.count, 3, 32, 32], scalars=bytes, device=device)
  
  let imageTensor = Tensor<Float>(images.permute([0, 2, 3, 1]))
  imageTensor /= 255.0
  if let mean = mean, let standardDeviation = standardDeviation then
    imageTensor = (imageTensor - mean) / standardDeviation

  
  let labels = Tensor (*<int32>*)(samples.map (fun x -> x.label), device=device)
  return LabeledImage(data: imageTensor, label: labels)

*)
