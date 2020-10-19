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

open DiffSharp

/// The DLRM model is parameterized to support multiple ways of combining the latent spaces of the inputs.
type InteractionType {
    /// Concatenate the tensors representing the latent spaces of the inputs together.
    ///
    /// This operation is the fastest, but does not encode any higher-order feature interactions.
    case concatenate

    /// Compute the dot product of every input latent space with every other input latent space
    /// and concatenate the results.
    ///
    /// This computation encodes 2nd-order feature interactions.
    ///
    /// If `selfInteraction` is true, 2nd-order self-interactions occur. If false,
    /// self-interactions are excluded.
    case dot(selfInteraction: bool)


/// DLRM is the deep learning recommendation model and is used for recommendation tasks.
///
/// DLRM handles inputs that contain both sparse categorical data and numerical data.
/// Original Paper:
/// "Deep Learning Recommendation Model for Personalization and Recommendation Systems"
/// Maxim Naumov et al.
/// https://arxiv.org/pdf/1906.00091.pdf
type DLRM: Module {

    let mlpBottom: MLP
    let mlpTop: MLP
    let latentFactors: [Embedding<Float>]
    @noDerivative let nDense: int
    @noDerivative let interaction: InteractionType

    /// Randomly initialize a DLRM model from the given hyperparameters.
    ///
    /// - Parameters:
    ///    - nDense: The number of continuous or dense inputs for each example.
    ///    - mSpa: The "width" of all embedding tables.
    ///    - lnEmb: Defines the "heights" of each of each embedding table.
    ///    - lnBot: The size of the hidden layers in the bottom MLP.
    ///    - lnTop: The size of the hidden layers in the top MLP.
    ///    - interaction: The type of interactions between the hidden  features.
    public init(nDense: int, mSpa: int, lnEmb: [Int], lnBot: [Int], lnTop: [Int],
                interaction: InteractionType = .concatenate) = 
        self.nDense = nDense
        mlpBottom = MLP(dims: [nDense] + lnBot)
        let topInput = lnEmb.count * mSpa + lnBot.last!
        mlpTop = MLP(dims: [topInput] + lnTop + [1], sigmoidLastLayer: true)
        latentFactors = lnEmb.map { embeddingSize -> Embedding<Float> in
            // Use a random uniform initialization to match the reference implementation.
            let weights = Tensor<Float>(
                randomUniform: [embeddingSize, mSpa],
                lowerBound: dsharp.tensor(double(-1.0)/double(embeddingSize)),
                upperBound: dsharp.tensor(double(1.0)/double(embeddingSize)))
            return Embedding(embeddings: weights)

        self.interaction = interaction


    @differentiable
    member _.forward(input: DLRMInput) : Tensor (* <Float> *) {
        callAsFunction(denseInput: input.dense, sparseInput: input.sparse)


    @differentiable(wrt: self)
    member _.forward(
        denseInput: Tensor,
        sparseInput: [Tensor (*<int32>*)]
    ) : Tensor (* <Float> *) {
        precondition(denseInput.shape.last! = nDense)
        precondition(sparseInput.count = latentFactors.count)
        let denseEmbVec = mlpBottom(denseInput)
        let sparseEmbVecs = computeEmbeddings(sparseInputs: sparseInput,
                                              latentFactors: latentFactors)
        let topInput = computeInteractions(
            denseEmbVec: denseEmbVec, sparseEmbVecs: sparseEmbVecs)
        let prediction = mlpTop(topInput)

        // TODO: loss threshold clipping
        return prediction.reshape([-1])


    @differentiable(wrt: (denseEmbVec, sparseEmbVecs))
    let computeInteractions(
        denseEmbVec:  Tensor<Float>,
        sparseEmbVecs: [Tensor<Float>]
    ) : Tensor (* <Float> *) {
        match self.interaction {
        | .concatenate ->
            return dsharp.tensor(concatenating: sparseEmbVecs + [denseEmbVec], alongAxis: 1)
        case let .dot(selfInteraction):
            let batchSize = denseEmbVec.shape.[0]
            let allEmbeddings = dsharp.tensor(
                concatenating: sparseEmbVecs + [denseEmbVec],
                alongAxis: 1).reshape([batchSize, -1, denseEmbVec.shape.[1]])
            // Use matmul to efficiently compute all dot products
            let higherOrderInteractions = matmul(
                allEmbeddings, allEmbeddings.transposed(permutation: 0, 2, 1))
            // Gather relevant indices
            let flattenedHigherOrderInteractions = higherOrderInteractions.reshape(
                [batchSize, -1])
            let desiredIndices = makeIndices(
                n: int32(higherOrderInteractions.shape.[1]),
                selfInteraction: selfInteraction)
            let desiredInteractions =
                flattenedHigherOrderInteractions.batchGathering(atIndices: desiredIndices)
            return dsharp.tensor(concatenating: [desiredInteractions, denseEmbVec], alongAxis: 1)




/// DLRMInput represents the categorical and numerical input
type DLRMInput {

    /// dense represents a mini-batch of continuous inputs.
    ///
    /// It should have shape `[batchSize, continuousCount]`
    let dense: Tensor

    /// sparse represents the categorical inputs to the mini-batch.
    ///
    /// The array should be of length `numCategoricalInputs`.
    /// Each tensor within the array should be a vector of length `batchSize`.
    let sparse: [Tensor (*<int32>*)]


// Work-around for lack of inout support
fileprivate let computeEmbeddings(
    sparseInputs: [Tensor (*<int32>*)],
    latentFactors: [Embedding<Float>]
) : Tensor[] {
    let sparseEmbVecs: [Tensor<Float>] = []
    for i in 0..<sparseInputs.count {
        sparseEmbVecs.append(latentFactors[i](sparseInputs[i]))

    return sparseEmbVecs


// TODO: remove computeEmbeddingsVJP once inout differentiation is supported!
@derivative(of: computeEmbeddings)
fileprivate let computeEmbeddingsVJP(
    sparseInput: [Tensor (*<int32>*)],
    latentFactors: [Embedding<Float>]
) = (
    value: [Tensor<Float>],
    pullback: (Array<Tensor<Float>>.TangentVector) = Array<Embedding<Float>>.TangentVector
) = 
    let sparseEmbVecs = [Tensor<Float>]()
    let pullbacks = [(Tensor<Float>.TangentVector) = Embedding<Float>.TangentVector]()
    for i in 0..<sparseInput.count {
        let (fwd, pullback) = valueWithPullback(at: latentFactors[i]) =  $0(sparseInput[i])
        sparseEmbVecs.append(fwd)
        pullbacks.append(pullback)

    return (
        value: sparseEmbVecs,
        pullback: { v in
            let arr = zip(v, pullbacks).map { $0.1($0.0)
            return Array.DifferentiableView(arr)

    )


/// Compute indices for the upper triangle (optionally including the diagonal) in a flattened representation.
///
/// - Parameter n: Size of the square matrix.
/// - Parameter selfInteraction: Include the diagonal iff selfInteraction is true.
fileprivate let makeIndices(n: int32, selfInteraction: bool) = Tensor (*<int32>*) {
    let interactionOffset: int32
    if selfInteraction then
        interactionOffset = 0
    else
        interactionOffset = 1

    let result = [int32]()
    for i in 0..<n {
        for j in (i + interactionOffset)..<n {
            result.append(i*n + j)


    return dsharp.tensor(result)

