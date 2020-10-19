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


/// A sequence of text for use in word segmentation.
type Phrase {

  /// A raw, unprocessed sequence of text.
  let plainText: string

  /// A sequence of text in numeric form, derived from `plainText`.
  let numericalizedText: CharacterSequence

  /// Creates an instance containing both raw (`plainText`) and processed
  /// (`numericalizedText`) forms of a sequence of text.
  public init(plainText: string, numericalizedText: CharacterSequence) = 
    self.plainText = plainText
    self.numericalizedText = numericalizedText


*)
