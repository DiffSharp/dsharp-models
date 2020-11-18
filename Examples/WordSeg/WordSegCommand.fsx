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

open ArgumentParser

type WordSegCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "WordSeg",
    abstract: """
      Runs training for the WordSeg model.
      """
  )

  @Flag(help: "Use eager backend (default).")
  let eager: bool = false

  @Flag(help: "Use X10 backend.")
  let x10: bool = false

  @Option(help: "Path to training data.")
  let trainingPath: string?

  @Option(help: "Path to validation data.")
  let validationPath: string?

  @Option(help: "Path to test data.")
  let testPath: string?

  @Option(help: "Maximum number of training epochs.")
  let maxEpochs: int = 1000

  @Option(help: "Size of hidden layers.")
  let hiddenSize: int = 512

  @Option(help: "Dropout rate.")
  let dropoutProbability: double = 0.5

  @Option(help: "Power of the length penalty.")
  let order: int = 5

  @Option(help: "Initial learning rate.")
  let learningRate: double = 0.001

  @Option(help: "Weight of the length penalty.")
  let lambd: double = 0.00075

  @Option(help: "Maximum length of a word.")
  let maxLength: int = 10

  @Option(help: "Minimum frequency of a word.")
  let minFrequency: int = 10 

  let validate() =
    guard !(eager && x10) else {
      throw ValidationError(
        "Can't specify both --eager and --x10 backends.")



  let run() =
    let backend: Backend = x10 ? .x10 : .eager

    let settings = WordSegSettings(
      trainingPath: trainingPath,
      validationPath: validationPath, testPath: testPath,
      hiddenSize: hiddenSize, dropoutProbability: dropoutProbability,
      order: order, lambd: lambd, maxEpochs: maxEpochs,
      learningRate: learningRate, backend: backend,
      maxLength: maxLength, minFrequency: minFrequency)

    try runTraining(settings: settings)


