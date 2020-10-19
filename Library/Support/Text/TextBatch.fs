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

namespace Support

open DiffSharp

/// Tokenized text passage.
type TextBatch(tokenIds: Tensor (*<int32>*), tokenTypeIds: Tensor (*<int32>*), mask: Tensor (*<int32>*)) =
  /// IDs that correspond to the vocabulary used while tokenizing.
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  member _.tokenIds = tokenIds

  /// IDs of the token types (e.g., sentence A and sentence B in BERT).
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  member _.tokenTypeIds = tokenTypeIds 

  /// Mask over the sequence of tokens specifying which ones are "real" as 
  /// opposed to "padding".
  /// The shape of this tensor is `[batchSize, maxSequenceLength]`.
  member _.mask = mask 

//// TODO: Use parallelism to grab the samples in parallel.
//extension TextBatch: Collatable {
//  /// Creates an instance from collating `samples`.
//  public init<BatchSamples: Collection>(collating samples: BatchSamples)
//  where BatchSamples.Element = Self {
//    self.init(
//      tokenIds: .init(concatenating: samples.map (fun x -> x.tokenIds)), 
//      tokenTypeIds: .init(concatenating: samples.map (fun x -> x.tokenTypeIds)), 
//      mask: .init(concatenating: samples.map (fun x -> x.mask))
//    )
//
//


//module TextBatch = 
//    extension Collection where Element = TextBatch {
//  /// Returns the elements of `self`, padded to `maxLength` if specified
//  /// or the maximum length of the elements in `self` otherwise.
//  let paddedAndCollated(to maxLength: int? = nil, on device: Device = .default) = TextBatch {
//    let maxLength = maxLength ?? self.map { $0.tokenIds.shape.[1].max()!
//    let paddedTexts = self.map { text -> TextBatch in
//      let paddingSize = maxLength - text.tokenIds.shape.[1]
//      return TextBatch(
//        tokenIds: dsharp.tensor(copying: text.tokenIds.padded(forSizes: [
//          (before: 0, after: 0),
//          (before: 0, after: paddingSize)]), device),
//        tokenTypeIds: dsharp.tensor(copying: text.tokenTypeIds.padded(forSizes: [
//          (before: 0, after: 0),
//          (before: 0, after: paddingSize)]), device),
//        mask: dsharp.tensor(copying: text.mask.padded(forSizes: [
//          (before: 0, after: 0),
//          (before: 0, after: paddingSize)]), device))
//

//    if count = 1 then return paddedTexts.first!
//    return paddedTexts.collated
//
//