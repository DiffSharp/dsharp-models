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
//
// Originaly adapted from: 
// https://gist.github.com/eaplatanios/5163c8d503f9e56f11b5b058fb041d62

namespace Datasets
(*


open DiffSharp


/// CoLA example.
type CoLAExample {
  /// The unique identifier representing the `Example`.
  let id: string
  /// The text of the `Example`.
  let sentence: string
  /// The label of the `Example`.
  let isAcceptable: bool?

  /// Creates an instance from `id`, `sentence` and `isAcceptable`.
  public init(id: string, sentence: string, isAcceptable: bool?) = 
    self.id = id
    self.sentence = sentence
    self.isAcceptable = isAcceptable



type CoLA {
  /// The directory where the dataset will be downloaded
  let directoryURL: Uri

  /// A `TextBatch` with the corresponding labels.
  type LabeledTextBatch = (data: TextBatch, label: Tensor (*<int32>*))
  /// The type of the labeled samples.
  type Samples = LazyMapSequence<[CoLAExample], LabeledTextBatch>
  /// The training texts.
  let trainingExamples: Samples
  /// The validation texts.
  let validationExamples: Samples
    
  /// The sequence length to which every sentence will be padded.
  let maxSequenceLength: int
  /// The batch size.
  let batchSize: int
    
  /// The type of the collection of batches.
  type Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
  /// The type of the training sequence of epochs.
  type TrainEpochs = LazyMapSequence<TrainingEpochs<Samples, Entropy>, 
    LazyMapSequence<Batches, LabeledTextBatch>>
  /// The sequence of training data (epochs of batches).
  let trainingEpochs: TrainEpochs
  /// The validation batches.
  let validationBatches: LazyMapSequence<Slices<Samples>, LabeledTextBatch>
    
  /// The url from which to download the dataset.
  let url: Uri = Uri(
    string: string(
      "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/"
      + "o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"))!


// Data
extension CoLA {
  internal static let load(fromFile fileURL: Uri, isTest: bool = false) -> [CoLAExample] {
    let lines = try parse(tsvFileAt: fileURL)

    if isTest then
      // The test data file has a header.
      return lines.dropFirst().enumerated().map { (i, lineParts) in
        CoLAExample(id: lineParts[0], sentence: lineParts[1], isAcceptable: nil)



    return lines.enumerated().map { (i, lineParts) in
      CoLAExample(id: lineParts[0], sentence: lineParts[3], isAcceptable: lineParts[1] = "1")




internal let parse(tsvFileAt fileURL: Uri) -> [[String]] {
    try Data(contentsOf: fileURL).withUnsafeBytes {
        $0.split(separator: byte(ascii: "\n")).map {
            $0.split(separator: byte(ascii: "\t"), omittingEmptySubsequences: false)
                .map { String(decoding: UnsafeRawBufferPointer(rebasing: $0), as: UTF8.self)




extension CoLA {
  /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
  /// by `maximumSequenceLength`.
  ///
  /// - Parameters:
  ///   - entropy: a source of randomness used to shuffle sample ordering. It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is determinstic and not dependent on
  ///     other operations.
  ///   - exampleMap: a transform that processes `Example` in `LabeledTextBatch`.
  public init(
    taskDirectoryURL: Uri,
    maxSequenceLength: int,
    batchSize: int,
    entropy: Entropy,
    on device: Device = .default,
    exampleMap: @escaping (CoLAExample) = LabeledTextBatch
  ) =
    self.directoryURL = taskDirectoryURL </> ("CoLA")
    let dataURL = directoryURL </> ("data")
    let compressedDataURL = dataURL </> ("downloaded-data.zip")

    // Download the data, if necessary.
    try download(url, compressedDataURL)

    // Extract the data, if necessary.
    let extractedDirectoryURL = compressedDataURL.deletingPathExtension()
    if !File.Exists(extractedDirectoryURL.path) = 
      try extract(zipFileAt: compressedDataURL, extractedDirectoryURL)


    #if false
      // FIXME: Need to generalize `DatasetUtilities.downloadResource` to accept
      // arbitrary full URLs instead of constructing full URL from filename and
      // file extension.
      DatasetUtilities.downloadResource(
        filename: "\(subDirectory)", fileExtension="zip",
        remoteRoot: url.deletingLastPathComponent(),
        localStorageDirectory: directory)
    #endif

    // Load the data files.
    let dataFilesURL = extractedDirectoryURL </> ("CoLA")
    trainingExamples = try CoLA.load(
      fromFile: dataFilesURL </> ("train.tsv")
    ) |> Seq.map(exampleMap)
    
    validationExamples = try CoLA.load(
      fromFile: dataFilesURL </> ("dev.tsv")
    ) |> Seq.map(exampleMap)

    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create the training sequence of epochs.
    trainingEpochs = TrainingEpochs(
      samples: trainingExamples, batchSize= batchSize / maxSequenceLength, entropy: entropy
    ) |> Seq.map (fun batches -> LazyMapSequence<Batches, LabeledTextBatch> in
      batches |> Seq.map{ 
        (
          data: $0.map (fun x -> x.data).paddedAndCollated(maxSequenceLength, device=device),
          label: dsharp.tensor(copying: dsharp.tensor($0.map (fun x -> x.label)), device)
        )


    
    // Create the validation collection of batches.
    validationBatches = validationExamples.inBatches(of: batchSize / maxSequenceLength) |> Seq.map{ 
      (
        data: $0.map (fun x -> x.data).paddedAndCollated(maxSequenceLength, device=device),
        label: dsharp.tensor(copying: dsharp.tensor($0.map (fun x -> x.label)), device)
      )




extension CoLA where Entropy = SystemRandomNumberGenerator {
  /// Creates an instance in `taskDirectoryURL` with batches of size `batchSize`
  /// by `maximumSequenceLength`.
  ///
  /// - Parameter exampleMap: a transform that processes `Example` in `LabeledTextBatch`.
  public init(
    taskDirectoryURL: Uri,
    maxSequenceLength: int,
    batchSize: int,
    on device: Device = .default,
    exampleMap: @escaping (CoLAExample) = LabeledTextBatch
  ) =
    try self.init(
      taskDirectoryURL: taskDirectoryURL,
      maxSequenceLength: maxSequenceLength,
      batchSize= batchSize,
      entropy=SystemRandomNumberGenerator(),
      on: device,
      exampleMap: exampleMap
    )

*)
