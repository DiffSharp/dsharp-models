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

module Models.ImageClassification.DLRM

open System.Diagnostics
open DiffSharp
open DiffSharp.Model
open Models.ImageClassification.MLP

/// The DLRM model is parameterized to support multiple ways of combining the latent spaces of the inputs.
type InteractionType =
    /// Concatenate the tensors representing the latent spaces of the inputs together.
    ///
    /// This operation is the fastest, but does not encode any higher-order feature interactions.
    | Concatenate

    /// Compute the dot product of every input latent space with every other input latent space
    /// and concatenate the results.
    ///
    /// This computation encodes 2nd-order feature interactions.
    ///
    /// If `selfInteraction` is true, 2nd-order self-interactions occur. If false,
    /// self-interactions are excluded.
    | Dot of selfInteraction: bool

/// DLRMInput represents the categorical and numerical input
type DLRMInput(dense: Tensor, sparse: Tensor[]) =

    /// dense represents a mini-batch of continuous inputs.
    ///
    /// It should have shape `[batchSize, continuousCount]`
    member _.dense = dense

    /// sparse represents the categorical inputs to the mini-batch.
    ///
    /// The array should be of length `numCategoricalInputs`.
    /// Each tensor within the array should be a vector of length `batchSize`.
    member _.sparse = sparse

/// DLRM is the deep learning recommendation model and is used for recommendation tasks.
///
/// DLRM handles inputs that contain both sparse categorical data and numerical data.
/// Original Paper:
/// "Deep Learning Recommendation Model for Personalization and Recommendation Systems"
/// Maxim Naumov et al.
/// https://arxiv.org/pdf/1906.00091.pdf
///
/// Randomly initialize a DLRM model from the given hyperparameters.
///
/// - Parameters:
///    - nDense: The number of continuous or dense inputs for each example.
///    - mSpa: The "width" of all embedding tables.
///    - lnEmb: Defines the "heights" of each of each embedding table.
///    - lnBot: The size of the hidden layers in the bottom MLP.
///    - lnTop: The size of the hidden layers in the top MLP.
///    - interaction: The type of interactions between the hidden  features.
type DLRM(nDense: int, mSpa: int, lnEmb: int[], lnBot: int[], lnTop: int[], ?interaction: InteractionType) = 
    inherit Model<DLRMInput, Tensor>() 
    let interaction = defaultArg interaction Concatenate
    let mlpBottom = MLP(dims= [| yield nDense; yield! lnBot |])
    let topInput = lnEmb.Length * mSpa + (lnBot |> Array.last)
    let mlpTop = MLP(dims= [| yield topInput; yield! lnTop; yield 1 |], sigmoidLastLayer=true)
    let latentFactors = 
        lnEmb |> Array.map (fun embeddingSize -> 
            // Use a random uniform initialization to match the reference implementation.
            let weights = 
                dsharp.rand([| embeddingSize; mSpa |],
                    low=dsharp.tensor(double(-1.0)/double(embeddingSize)),
                    high=dsharp.tensor(double(1.0)/double(embeddingSize)))
            Embedding(embeddings=weights))


    /// Compute indices for the upper triangle (optionally including the diagonal) in a flattened representation.
    ///
    /// - Parameter n: Size of the square matrix.
    /// - Parameter selfInteraction: Include the diagonal iff selfInteraction is true.
    let makeIndices(n: int32, selfInteraction: bool) = 
        let interactionOffset = if selfInteraction then 0 else 1
        dsharp.tensor([| for i in 0..n-1 do for j in (i + interactionOffset)..n-1 -> i*n + j |], dtype=Dtype.Int32)

    let computeInteractions(denseEmbVec: Tensor, sparseEmbVecs: Tensor[]) : Tensor =
        match interaction with
        | Concatenate ->
            dsharp.cat([| yield! sparseEmbVecs; yield denseEmbVec |], dim=1)
        | Dot(selfInteraction) ->
            let batchSize = denseEmbVec.shape.[0]
            let allEmbeddings = dsharp.cat([| yield! sparseEmbVecs; yield denseEmbVec |],dim=1).view([batchSize; -1; denseEmbVec.shape.[1]])
            // Use matmul to efficiently compute all dot products
            let higherOrderInteractions = dsharp.matmul(allEmbeddings, allEmbeddings.permute( [| 0; 2; 1 |]))// Gather relevant indices
            let flattenedHigherOrderInteractions = higherOrderInteractions.view([batchSize; -1])
            let desiredIndices = makeIndices(int32(higherOrderInteractions.shape.[1]), selfInteraction)
            let desiredInteractions = failwith "TBD" // flattenedHigherOrderInteractions.batchGathering(desiredIndices)
            dsharp.cat ([desiredInteractions; denseEmbVec], dim=1)

    // Work-around for lack of inout support
    let computeEmbeddings(sparseInputs: Tensor[], latentFactors: Embedding[]) =
        [| for i in 0..sparseInputs.Length-1 do
            latentFactors.[i].[sparseInputs.[i]] |]

    //// TODO: remove computeEmbeddingsVJP once inout differentiation is supported!
    //let computeEmbeddingsVJP(sparseInput: Tensor[], latentFactors: Embedding[]) = 
    //    let sparseEmbVecs = [Tensor]()
    //    let pullbacks = [(Tensor.TangentVector) = Embedding.TangentVector]()
    //    for i in 0..sparseInput.Length-1 do
    //        let (fwd, pullback) = valueWithPullback(at: latentFactors.[i]) =  $0(sparseInput[i])
    //        sparseEmbVecs.append(fwd)
    //        pullbacks.append(pullback)

    //    (
    //        value: sparseEmbVecs,
    //        pullback: { v in
    //            let arr = zip(v, pullbacks).map (fun x -> x.1($0.0))
    //            Array.DifferentiableView(arr)

    //    )

    override _.forward(input: DLRMInput) : Tensor =
        let denseInput = input.dense
        let sparseInput = input.sparse

        Debug.Assert(denseInput.shape |> Array.last = nDense)
        Debug.Assert(sparseInput.Length = latentFactors.Length)
        let denseEmbVec = mlpBottom.forward(denseInput)
        let sparseEmbVecs = computeEmbeddings(sparseInput, latentFactors)
        let topInput = computeInteractions(denseEmbVec, sparseEmbVecs)
        let prediction = mlpTop.forward(topInput)

        // TODO: loss threshold clipping
        prediction.view([-1])


