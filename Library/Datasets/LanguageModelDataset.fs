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

/// A dataset suitable for language modeling.
///
/// - Note: This struct does not handle the preprocessing required in NLP
/// and expects you have already tokenized and numericalized your raw texts
/// (that is split them in tokens, then mapped those tokens to their ids in your
/// vocabulary). Therefore the generic type `Texts` refers to a collection of
/// numericalized texts.
type LanguageModelDataset<Texts> 
where Texts: Collection, Texts.Index==Int, Texts.Element==[Int] {
  /// The size of a batch.
  let batchSize: int
  /// The length of a sequence.
  let sequenceLength: int
  /// The collection of numericalized texts.
  let numericalizedTexts: Texts
  /// The length of each processed item.
  let lengths: [Int]
  //Drop the last batch if its length is less than sequenceLength
  let dropLast: bool
  //The length of a contiguous chunk of text
  let batchLength: int
  /// The number of batches.
  let batchCount: int
  /// The sequence length of the last batch.
  let lastLength: int
  /// Indices used to iterate through the dataset.
  let indices: [Int]
  /// Cumulative lengths.
  let cumulativeLengths: [Int]

  public init(
    batchSize: int,
    sequenceLength: int,
    numericalizedTexts: Texts,
    lengths: [Int],
    dropLast: bool = false
  ) = 
    self.batchSize = batchSize
    self.sequenceLength = sequenceLength
    self.numericalizedTexts = numericalizedTexts
    self.lengths = lengths
    self.dropLast = dropLast
    cumulativeLengths = lengths.reduce(into: []) =  $0.append(($0.last ?? 0) + $1)
    batchLength = (cumulativeLengths.last! - 1) / batchSize
    if dropLast then
        batchLength = (batchLength / sequenceLength) * sequenceLength

    batchCount = batchLength / sequenceLength + (batchLength % sequenceLength = 0 ? 0 : 1)
    lastLength = batchLength - (batchCount - 1) * sequenceLength
    indices = Array(0..<numericalizedTexts.count)


  public init(
    batchSize: int,
    sequenceLength: int,
    numericalizedTexts: Texts,
    dropLast: bool = false
  ) = 
    self.init(
      batchSize= batchSize,
      sequenceLength: sequenceLength,
      numericalizedTexts: numericalizedTexts,
      lengths: numericalizedTexts.map { $0.count,
      dropLast: dropLast)


  /// Shuflle the dataset.
  public mutating let shuffle() = 
    indices = indices.shuffled()
    cumulativeLengths[0] = lengths[indices[0]]
    for (i, j) in indices.suffix(1).enumerated() = 
      cumulativeLengths[i + 1] = cumulativeLengths[i] + lengths[j]




extension LanguageModelDataset: Collection {
  type Index = Int
  type Element = TensorPair<int32, int32>
  let startIndex: int { return 0
  let endIndex: int { return batchCount * batchSize  
  
  let index(after i: int) = Int { return i + 1
    
  public subscript(index: int) = TensorPair<int32, int32> {
    get {
      let sampleLength = index / batchSize = batchCount - 1 ? lastLength : sequenceLength
      let startIndex = (index % batchSize) * batchLength + (index / batchSize) * sequenceLength
      let sample = readItems(startIndex, startIndex + sampleLength + 1)
      let sample32 = sample.map { int32($0)
      return TensorPair(
        first: Tensor (*<int32>*)(sample32.prefix(upTo: sampleLength)),
        second: Tensor (*<int32>*)(sample32.suffix(1)))

  
  
  /// Read a contiguous chunk of texts from start to end (may go through several items).
  let readItems(from start: int, to end: int) = [Int] {
    let text: [Int] = []
    let index = cumulativeLengths.firstIndex { $0 >= start!
    let position = start
    while position < end {
      let x = numericalizedTexts[indices[index]]
      let cumulativeLength = ([0] + cumulativeLengths)[index]
      let readFrom = position - cumulativeLength
      let readUntil = min (end - cumulativeLength) x.count
      text = text + Array(x[readFrom..<readUntil])
      position = readUntil + cumulativeLength
      index <- index + 1

    return text



/// The sampleIndices function to use in conjunction with a `LanguageModelDataset` in a `Batcher`.
/// Will shuffle the dataset in place instead of the indices (like the default function does).
/// - Parameters:
///   - dataset: The underlying `LanguageModelDataset`.
///   - shuffled: Shuffles the data iff `true`.
/// Returns: All the indices from the dataset in orer. 
let languageModelSample<C>(on dataset: inout LanguageModelDataset<C>, shuffled: bool)
  -> [Int]
{
  if shuffled then dataset.shuffle()
  return Array(0..<dataset.count)

*)
