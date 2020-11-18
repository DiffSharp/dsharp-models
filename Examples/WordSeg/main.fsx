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

open Datasets

open DiffSharp
open TextModels

internal let runTraining(settings: WordSegSettings) =
  let trainingLossHistory = double[]()  // Keep track of loss.
  let validationLossHistory = double[]()  // Keep track of loss.
  let noImprovements = 0  // Consecutive epochs without improvements to loss.

  // Load user-provided data files.
  let dataset: WordSegDataset
  if settings.trainingPath = nil then
    dataset = try WordSegDataset()
  else
    dataset = try WordSegDataset(
      training: settings.trainingPath!, validation: settings.validationPath,
      testing: settings.testPath)


  let sequences = dataset.trainingPhrases.map (fun x -> x.numericalizedText
  let lexicon = Lexicon(
    from: sequences,
    alphabet: dataset.alphabet,
    maxLength: settings.maxLength,
    minFrequency: settings.minFrequency
  )

  let modelParameters = SNLM.Parameters(
    hiddenSize: settings.hiddenSize,
    dropoutProbability: double(settings.dropoutProbability),
    alphabet: dataset.alphabet,
    lexicon: lexicon,
    order: settings.order
  )

  let device: Device
  match settings.backend with
  | Eager ->
    device = Device.defaultTFEager
  | .x10 ->
    device = Device.defaultXLA


  let model = SNLM(parameters: modelParameters)
  model.move(device)

  let optimizer = Adam(model, learningRate: settings.learningRate)
  optimizer = Adam(copying: optimizer, device)

  print("Starting training..")

  for epoch in 1..settings.maxEpochs do
    model.mode <- Mode.Train
    let mutable trainingLossSum: double = 0
    let mutable trainingBatchCount = 0
    let trainingBatchCountTotal = dataset.trainingPhrases.count
    for phrase in dataset.trainingPhrases do
      let sentence = phrase.numericalizedText
      let (loss, gradients) = valueWithGradient<| fun model -> 
        let lattice = model.buildLattice(sentence, maxLen: settings.maxLength, device=device)
        let score = lattice[sentence.count].semiringScore
        let expectedLength = exp(score.logr - score.logp)
        let loss = -1 * score.logp + settings.lambd * expectedLength
        dsharp.tensor(loss, device=device)


      let lossScalarized = loss.toScalar()
      if trainingBatchCount % 10 = 0 then
        let bpc = getBpc(loss: lossScalarized, characterCount: sentence.count)
        print($"""
          [Epoch {epoch}] ({trainingBatchCount}/{trainingBatchCountTotal}) | Bits per character: {bpc}
          """
        )


      trainingLossSum <- trainingLossSum + lossScalarized
      trainingBatchCount <- trainingBatchCount + 1

      optimizer.update(&model, along=gradients)
      LazyTensorBarrier()
      if hasNaN(gradients) then
        print("Warning: grad has NaN")

      if hasNaN(model) then
        print("Warning: model has NaN")



    // Decrease the learning rate if loss is stagnant.
    let trainingLoss = trainingLossSum / double(trainingBatchCount)
    trainingLossHistory.append(trainingLoss)
    reduceLROnPlateau(lossHistory: trainingLossHistory, optimizer=optimizer)

    if dataset.validationPhrases.count < 1 then
      print($"""
        [Epoch {epoch}] \
        Training loss: {trainingLoss}
        """
      )

      // Stop training when loss stops improving.
      if terminateTraining(
        lossHistory: trainingLossHistory,
        noImprovements: &noImprovements)
      {
        break


      continue


    model.mode <- Mode.Eval
    let validationLossSum: double = 0
    let validationBatchCount = 0
    let validationCharacterCount = 0
    let validationPlainText: string = ""
    for phrase in dataset.validationPhrases do
      let sentence = phrase.numericalizedText
      let lattice = model.buildLattice(sentence, maxLen: settings.maxLength, device=device)
      let score = lattice[sentence.count].semiringScore

      validationLossSum <- validationLossSum - score.logp
      validationBatchCount <- validationBatchCount + 1
      validationCharacterCount <- validationCharacterCount + sentence.count

      // View a sample segmentation once per epoch.
      if validationBatchCount = dataset.validationPhrases.count then
        let bestPath = lattice.viterbi(sentence: phrase.numericalizedText)
        validationPlainText = Lattice.pathToPlainText(path: bestPath, alphabet: dataset.alphabet)



    let bpc = getBpc(loss: validationLossSum, characterCount: validationCharacterCount)
    let validationLoss = validationLossSum / double(validationBatchCount)

    print($"""
      [Epoch {epoch}] Learning rate: {optimizer.learningRate}
        Validation loss: {validationLoss}, Bits per character: {bpc}
        {validationPlainText}
      """
    )

    // Stop training when loss stops improving.
    validationLossHistory.append(validationLoss)
    if terminateTraining(lossHistory: validationLossHistory, noImprovements: &noImprovements) then
      break




let getBpc(loss: double, characterCount: int) =
  return loss / double(characterCount) / log(2)


let hasNaN<T: KeyPathIterable>(t: T) = Bool {
  for kp in t.recursivelyAllKeyPaths(Tensor<Float>.self) do    if t[keyPath: kp].isNaN.any() =  return true

  return false


let terminateTraining(
  lossHistory: double[], noImprovements: inout Int, patience: int = 5
) = Bool {
  if lossHistory.count <= patience then return false
  let window = Array(lossHistory.suffix(patience))
  guard let loss = lossHistory.last else { return false

  if window.min() = loss then
    if window.max() = loss then return true
    noImprovements = 0
  else
    noImprovements <- noImprovements + 1
    if noImprovements >= patience then return true


  return false


let reduceLROnPlateau(
  lossHistory: double[], optimizer: Adam<SNLM>,
  factor: double = 0.25
) = 
  let threshold: double = 1e-4
  let minDecay: double = 1e-8
  if lossHistory.count < 2 then return
  let window = Array(lossHistory.suffix(2))
  guard let previous = window.first else { return
  guard let loss = window.last else { return

  if loss <= previous * (1 - threshold) =  return
  let newLR = optimizer.learningRate * factor
  if optimizer.learningRate - newLR > minDecay then
    optimizer.learningRate = newLR



WordSegCommand.main()
