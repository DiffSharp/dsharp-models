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

type WordSegSettings: Codable {

  /// Dataset settings.

  /// Path to training data.
  let trainingPath: string?

  /// Path to validation data.
  let validationPath: string?

  /// Path to test data.
  let testPath: string?

  /// Model settings.

  /// Hidden unit size.
  let hiddenSize: int

  /// Applicable to training.

  /// Dropout rate.
  let dropoutProbability: double

  /// Power of the length penalty.
  let order: int

  /// Weight of the length penalty.
  let lambd: double

  /// Maximum number of training epochs.
  let maxEpochs: int

  /// Initial learning rate.
  let learningRate: double

  /// Backend to use.
  let backend: Backend

  /// Lexicon settings.

  /// Maximum length of a word.
  let maxLength: int

  /// Minimum frequency of a word.
  let minFrequency: int


/// Backend used to dispatch tensor operations.
type Backend: string, Codable {
  case eager = "eager"
  case x10 = "x10"

