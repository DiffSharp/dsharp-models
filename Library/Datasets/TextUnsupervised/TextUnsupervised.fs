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
(*


open DiffSharp

type TextUnsupervisedVariant: string {
  /// - Source: [Einstein AI WikiText-103](
  ///             https://blog.einstein.ai/
  ///             the-wikitext-long-term-dependency-language-modeling-dataset/).
  case wikiText103 = "WikiText103"

  /// Default variant.
  /// - Source: [Einstein AI WikiText-2](
  ///             https://blog.einstein.ai/
  ///             the-wikitext-long-term-dependency-language-modeling-dataset/).
  case wikiText2 = "WikiText2"


private type ITextUnsupervisedVariantDetails {
  let variant=TextUnsupervisedVariant { get set
  let location: Uri { get set
  let trainingDirectoryName: string { get set
  let validationDirectoryName: string { get set
  let filename: string { get set
  let encodedFileName: string? { get set
  let fileExtension: string { get set


type TextUnsupervised {
  private struct WikiText103Details: TextUnsupervisedVariantDetails {
    let variant = TextUnsupervisedVariant.wikiText103
    let location = Uri("https://s3.amazonaws.com/fast-ai-nlp/")!
    let trainingDirectoryName = "train"
    let validationDirectoryName = "test"
    let filename = "wikitext-103"
    let encodedFileName: string? = nil
    let fileExtension = "tgz"


  private struct WikiText2Details: TextUnsupervisedVariantDetails {
    let variant = TextUnsupervisedVariant.wikiText2

    let location = Uri(
      string: "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/WikiText2/")!

    let trainingDirectoryName = "train"
    let validationDirectoryName = "test"
    let filename = "wikitext-2"
    let encodedFileName: string? = "wikitext-2-encoded"
    let fileExtension = "tgz"


  type Samples = LanguageModelDataset<[[Int]]>
  type LabeledTextBatch = LabeledData<Tensor, Tensor>

  type Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
  type Training = LazyMapSequence<
    TrainingEpochs<Samples, Entropy>, 
    LazyMapSequence<Batches, LabeledTextBatch>
  >

  type Validation = LazyMapSequence<
    Slices<Samples>, 
    LabeledTextBatch
  >

  let training: Training
  let validation: Validation

  let trainingDataset: Samples
  let validationDataset: Samples
  let bpe: BytePairEncoder?
  let variant=TextUnsupervisedVariant
  let variantDetails: TextUnsupervisedVariantDetails

  public init(
    bpe: BytePairEncoder? = nil,
    variant=TextUnsupervisedVariant = TextUnsupervisedVariant.wikiText2,
    trainingbatchSize: int = 8, validationbatchSize: int = 4, sequenceLength=int = 1024,
    trainingDocumentCount: int = 4, validationDocumentCount: int = 4,
    entropy: Entropy,
    on device: Device = Device.defaultTFEager
  ) = 
    try
      self.bpe = bpe

      self.variant = variant
      match variant with
      | .wikiText103 ->
        let variantDetails = WikiText103Details()
        self.variantDetails = variantDetails
      | .wikiText2 ->
        let variantDetails = WikiText2Details()
        self.variantDetails = variantDetails


      let localStorageDirectory: Uri = File.temporaryDirectory
         </> (
          variant.rawValue)
      self.trainingDataset = try TextUnsupervised.loadTraining(
        localStorageDirectory=localStorageDirectory, bpe: bpe,
        variantDetails: variantDetails, batchSize= trainingBatchSize,
        sequenceLength=sequenceLength, documentCount: trainingDocumentCount)
      self.validationDataset = try TextUnsupervised.loadValidation(
        localStorageDirectory=localStorageDirectory, bpe: bpe,
        variantDetails: variantDetails, batchSize= validationBatchSize,
        sequenceLength=sequenceLength, documentCount: validationDocumentCount)

      training = TrainingEpochs(
        samples: trainingDataset, 
        batchSize= trainingBatchSize, 
        entropy: entropy
      ) |> Seq.map (fun batches -> LazyMapSequence<Batches, LabeledTextBatch> in
        batches |> Seq.map {
          LabeledData(
            data: dsharp.tensor($0.map (fun x -> x.first).map { Tensor ($0.scalars, device=device, dtype=Dtype.int32)),
            label: dsharp.tensor($0.map (fun x -> x.second).map { Tensor ($0.scalars, device=device, dtype=Dtype.int32))
          )



      validation = validationDataset.inBatches(of: validationBatchSize) |> Seq.map {
        LabeledData(
          data: Tensor ($0.map (fun x -> x.first).map { Tensor ($0.scalars, device=device, dtype=Dtype.int32)),
          label: Tensor ($0.map (fun x -> x.second).map { Tensor ($0.scalars, device=device, dtype=Dtype.int32))
        )


    with
      fatalError($"Could not load dataset for {variant}: {error}")



  static member downloadIfNotPresent(
    directory: FilePath, variantDetails: TextUnsupervisedVariantDetails, downloadEncodedFile: bool
  ) = 
    let downloadPath = directory </> (variantDetails.variant.rawValue).path
    let directoryExists = File.Exists(downloadPath)
    let contentsOfDir = try? Directory.GetFiles(downloadPath)
    let directoryEmpty = (contentsOfDir = nil) || (contentsOfDir!.isEmpty)

    guard !directoryExists || directoryEmpty else { return

    // Downloads and extracts dataset files.
    let _ = DatasetUtilities.downloadResource(
      filename: (downloadEncodedFile ? variantDetails.encodedFileName! : variantDetails.filename),
      fileExtension: variantDetails.fileExtension,
      remoteRoot: variantDetails.location, localStorageDirectory: directory, extract: true)


  static member readCSV(in file: Uri) -> string[] {
    let rawText = try! String(contentsOf: file)
    let rows = rawText.components(separatedBy: "\"\n\"")
    rows[0] = String(rows[0].dropFirst())
    rows[rows.indices |> Array.last] = String(rows |> Array.last.dropLast(2))
    return rows


  static member readEncoded(in file: Uri) -> [Int] {
    let rawText = try! String(contentsOf: file)
    let rows = rawText.components(separatedBy: "\n")
    let tokens: int[] = Array()
    for row in rows do
      guard let encoded = int(row) else { continue
      tokens.append(encoded)

    return tokens


  static member embedding(for string: string, bpe: BytePairEncoder) = [Int] {
    let tokens = bpe.encode(token: string, variant=.gpt2)
    // TODO(michellecasbon): Decide how to prevent OOV or choose a better ID (probably not 0).
    let ids = tokens.map { bpe.vocabulary.id(forToken: $0) ?? 0
    return ids


  /// Returns a LanguageModelDataset by processing files specified by 'variantDetails' which
  /// resides in 'directory'.
  ///
  /// Download the files if not present. If bpe is nil which means skip bype pair encoding,
  /// then download the encoded file instead.
  ///
  /// - Parameter name= name of the dataset. Ususally 'train' or 'test'.
  /// - Parameter directory: directory that files are read from.
  /// - Parameter bpe: byte pair encoder used for encoding text.
  /// - Parameter variantDetails: an object containing information of filename, location, etc.
  /// - Parameter batchSize= number of sequences in a batch.
  /// - Parameter sequenceLength=number of characters in a sequence.
  /// - Parameter documentCount: number of documents to proceed. (Refer let readCSV() to see how
  ///   a text file is chunked into documents.)
  static member loadDirectory(
    named name: string, in directory: Uri, bpe: BytePairEncoder?,
    variantDetails: TextUnsupervisedVariantDetails, batchSize: int, sequenceLength=int,
    documentCount: int = 4
  ) -> LanguageModelDataset<[[Int]]> {
    Debug.Assert(
      bpe <> nil || variantDetails.encodedFileName <> nil,
      "bpe must be provided when encodedFileName is nil.")
    downloadIfNotPresent(
      directory, variantDetails: variantDetails, downloadEncodedFile: bpe = nil)

    let encodedDocs: [[Int]] = []
    if let bpe = bpe then
      let path = directory </> ($"{variantDetails.filename}/{name}.csv")
      let documentsFull = try readCSV(in: path)
      let documents = Array(documentsFull[0..<min(documentCount, documentsFull.count)])
      encodedDocs = documents.concurrentMap { embedding($0, bpe: bpe)
    else
      let pathPrefix = directory </> (
        $"\(variantDetails.encodedFileName!)/{name}").path
      encodedDocs = (0..<documentCount).map { Uri(fileURLWithPath= $"{pathPrefix}/doc_\($0).txt")
        .concurrentMap
      { try! readEncoded(in: $0)


    return LanguageModelDataset(
      batchSize= batchSize,
      sequenceLength=sequenceLength,
      numericalizedTexts: encodedDocs,
      lengths: encodedDocs.map (fun x -> x.count),
      dropLast: true
    )


  static member loadTraining(
    localStorageDirectory: Uri, bpe: BytePairEncoder?,
    variantDetails: TextUnsupervisedVariantDetails, batchSize: int, sequenceLength=int,
    documentCount: int
  )
   
    -> LanguageModelDataset<[[Int]]>
  {
    return try loadDirectory(
      named: variantDetails.trainingDirectoryName, in: localStorageDirectory, bpe: bpe,
      variantDetails: variantDetails, batchSize= batchSize, sequenceLength=sequenceLength,
      documentCount: documentCount)


  static member loadValidation(
    localStorageDirectory: Uri, bpe: BytePairEncoder?,
    variantDetails: TextUnsupervisedVariantDetails, batchSize: int, sequenceLength=int,
    documentCount: int
  )
   
    -> LanguageModelDataset<[[Int]]>
  {
    return try loadDirectory(
      named: variantDetails.validationDirectoryName, in: localStorageDirectory, bpe: bpe,
      variantDetails: variantDetails, batchSize= batchSize, sequenceLength=sequenceLength,
      documentCount: documentCount)



extension TextUnsupervised where Entropy = SystemRandomNumberGenerator {
  public init(
    bpe: BytePairEncoder? = nil,
    variant=TextUnsupervisedVariant = TextUnsupervisedVariant.wikiText2,
    trainingbatchSize: int = 8, validationbatchSize: int = 4, sequenceLength=int = 1024,
    trainingDocumentCount: int = 4, validationDocumentCount: int = 4,
    on device: Device = Device.defaultTFEager
  ) = 
    self.init(
      bpe: bpe,
      variant=variant,
      trainingbatchSize= trainingBatchSize,
      validationbatchSize= validationBatchSize,
      sequenceLength=sequenceLength,
      trainingDocumentCount: trainingDocumentCount,
      validationDocumentCount: validationDocumentCount,
      entropy=SystemRandomNumberGenerator(),
      on: device
    )



extension Array {
  let concurrentMap<B>(transform: @escaping (Element) = B) = [B] {
    let res = [B?](repeating: nil, count: count)
    let threadCount = min count 10
    let q = DispatchQueue(label: "sync queue")
    DispatchQueue.concurrentPerform(iterations: threadCount) =  threadId in
      for idx in stride(threadId, count, by: threadCount) = 
        let transformed = transform(self[idx])
        q.sync {
          res[idx] = transformed



    return res.map { $0!


*)
