// Copyright 2019 The TensorFlow Authors, adapted by the DiffSharp authors. All Rights Reserved.
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

#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r @"System.Runtime.Extensions.dll"
#load "Utilities.fsx"
#load "Attention.fsx"

open System.Diagnostics
open DiffSharp
open DiffSharp.Model
open DiffSharp.ShapeChecking
open global.Utilities
open global.Attention

type Linear with 
    member m.regularizationValue = 
        TangentVector 
            {| weight = m.weight.value; 
               bias = dsharp.tensor(0, device=m.bias.value.device) |}

type LayerNorm with 
    member m.regularizationValue = 
        TangentVector 
            {| offset= dsharp.tensor(0, device=m.offset.value.device)
               scale= dsharp.tensor(0, device=m.scale.value.device) |}

type Embedding with 
    member m.regularizationValue = 
        TangentVector 
            {| embeddings = m.embeddings |}

/// Input to a transformer layer.
type TransformerInput(sequence: Tensor, attentionMask: Tensor, ?batchSize: int) =
    /// Sequence that the transformer encoder operates over. The shape of this tensor is
    /// `[batchSize, sequenceLength, depth]` or `[batchSize, sequenceLength * depth]`.
    member _.sequence = sequence

    /// Mask to apply on the attention scores. This is a tensor with shape
    /// `[batchSize, sourceSequenceLength, targetSequenceLength]` or
    /// `[batchSize, sourceSequenceLength * targetSequenceLength]`. The values should be `1` or
    /// `0`. The attention scores will effectively be set to negative infinity for any positions in 
    /// the mask that are set to `0`, and will be unchanged for positions that are set to `1`.
    member _.attentionMask = attentionMask

    /// The batch size of this input. This is optional because it is only needed if the input
    /// sequences have been reshaped to matrices.
    member _.batchSize = batchSize

/// Transformer encoder layer.
///
/// - Source: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
///
/// Creates a transformer encoder layer.
///
/// - Parameters:
///   - hiddenSize: Size/depth of the transformer hidden representation.
///   - attentionHeadCount: Number of attention heads.
///   - attentionQueryActivation= Activation function applied to the attention query tensor.
///   - attentionKeyActivation= Activation function applied to the attention key tensor.
///   - attentionValueActivation= Activation function applied to the attention value tensor.
///   - intermediateSize: Size/depth of the transformer intermediate representation.
///   - intermediateActivation= Activation function applied to the intermediate representation.
///   - hiddenDropoutProbability: Dropout probability for the hidden representations.
///   - attentionDropoutProbability: Dropout probability for the attention scores.
///   - queryWeightInitializer: Initializer for the query transformation weight.
///   - queryBiasInitializer: Initializer for the query transformation bias.
///   - keyWeightInitializer: Initializer for the key transformation weight.
///   - keyBiasInitializer: Initializer for the key transformation bias.
///   - valueWeightInitializer: Initializer for the value transformation weight.
///   - valueBiasInitializer: Initializer for the value transformation bias.
///   - attentionWeightInitializer: Initializer for the attention transformation weight.
///   - attentionBiasInitializer: Initializer for the attention transformation bias.
///   - intermediateWeightInitializer: Initializer for the intermediate transformation weight.
///   - intermediateBiasInitializer: Initializer for the intermediate transformation bias.
///   - outputWeightInitializer: Initializer for the output transformation weight.
///   - outputBiasInitializer: Initializer for the output transformation bias.
type TransformerEncoderLayer(hiddenSize: int,
        attentionHeadCount: int,
        attentionQueryActivation: Activation,
        attentionKeyActivation: Activation,
        attentionValueActivation: Activation,
        intermediateSize: int,
        intermediateActivation: Activation,
        hiddenDropoutProbability: scalar,
        attentionDropoutProbability: scalar,
        ?queryWeightInitializer: ParameterInitializer,
        ?queryBiasInitializer: ParameterInitializer,
        ?keyWeightInitializer: ParameterInitializer,
        ?keyBiasInitializer: ParameterInitializer,
        ?valueWeightInitializer: ParameterInitializer,
        ?valueBiasInitializer: ParameterInitializer,
        ?attentionWeightInitializer: ParameterInitializer,
        ?attentionBiasInitializer: ParameterInitializer,
        ?intermediateWeightInitializer: ParameterInitializer,
        ?intermediateBiasInitializer: ParameterInitializer,
        ?outputWeightInitializer: ParameterInitializer,
        ?outputBiasInitializer: ParameterInitializer) =
    inherit Model()
    do 
        Debug.Assert(
            hiddenSize % attentionHeadCount = 0,
            "The hidden size of the transformer ({hiddenSize}) must be a multiple of the "
                + "attention head count ({attentionHeadCount}).")
    let multiHeadAttention = 
        MultiHeadAttention(
            sourceSize=hiddenSize,
            targetSize=hiddenSize,
            headCount=attentionHeadCount,
            headSize=hiddenSize / attentionHeadCount,
            queryActivation= attentionQueryActivation,
            keyActivation= attentionKeyActivation,
            valueActivation= attentionValueActivation,
            attentionDropoutProbability=attentionDropoutProbability,
            matrixResult=true,
            ?queryWeightInitializer=queryWeightInitializer,
            ?queryBiasInitializer=queryBiasInitializer,
            ?keyWeightInitializer=keyWeightInitializer,
            ?keyBiasInitializer=keyBiasInitializer,
            ?valueWeightInitializer=valueWeightInitializer,
            ?valueBiasInitializer=valueBiasInitializer)

    let attentionWeightInitializer = defaultArg attentionWeightInitializer defaultWeightInitializer
    let attentionBiasInitializer = defaultArg attentionBiasInitializer defaultBiasInitializer
    let intermediateWeightInitializer = defaultArg intermediateWeightInitializer defaultWeightInitializer
    let intermediateBiasInitializer = defaultArg intermediateBiasInitializer defaultBiasInitializer

    // TODO: Make dropout generic over the probability type.
    let hiddenDropout = Dropout(hiddenDropoutProbability.toDouble())
    let attentionWeight = attentionWeightInitializer(Shape [attentionHeadCount * hiddenSize / attentionHeadCount; hiddenSize])
    let attentionBias = attentionBiasInitializer(Shape [hiddenSize])
    let attentionLayerNorm = LayerNorm( featureCount=hiddenSize, axis= -1)
    let intermediateWeight = intermediateWeightInitializer(Shape [hiddenSize; intermediateSize])
    let intermediateBias = intermediateBiasInitializer(Shape [intermediateSize])
    let outputWeight = intermediateWeightInitializer(Shape [intermediateSize; hiddenSize])
    let outputBias = intermediateBiasInitializer(Shape [hiddenSize])
    let outputLayerNorm = LayerNorm(featureCount=hiddenSize, axis = -1)

    member _.regularizationValue =
        TangentVector
            {| multiHeadAttention=multiHeadAttention.regularizationValue
               attentionWeight=attentionWeight
               attentionBias=dsharp.tensor(0, device=attentionBias.device)
               attentionLayerNorm=attentionLayerNorm.regularizationValue
               intermediateWeight=intermediateWeight
               intermediateBias=dsharp.tensor(0, device=intermediateBias.device)
               outputWeight=outputWeight
               outputBias=dsharp.tensor(0, device=outputBias.device)
               outputLayerNorm=outputLayerNorm.regularizationValue |}

    override _.forward(input: Tensor) : Tensor =
        let input = Unchecked.defaultof<TransformerInput> // TBD
        let attentionInput =
            AttentionInput(
                source=input.sequence,
                target=input.sequence,
                mask=input.attentionMask,
                ?batchSize= input.batchSize)
        let attentionOutput = multiHeadAttention.forward(failwith "tbd" (* attentionInput *) )

        // Run a linear projection of `hiddenSize` and then add a residual connection to the input.
        let attentionOutput = dsharp.matmul(attentionOutput, attentionWeight) + attentionBias
        let attentionOutput = hiddenDropout.forward(attentionOutput)
        let attentionOutput = attentionLayerNorm.forward(attentionOutput + input.sequence)

        // The activation is only applied to the "intermediate" hidden layer.
        let intermediateOutput = dsharp.matmul(attentionOutput, intermediateWeight) + intermediateBias
        let intermediateOutput = intermediateActivation(intermediateOutput)

        // Project back to `hiddenSize` and add the residual.
        let output = dsharp.matmul(intermediateOutput, outputWeight) + outputBias
        let output = hiddenDropout.forward(output)
        let output = outputLayerNorm.forward(output + attentionOutput)

        output

/// Multi-headed and multi-layer transformer encoder.
///
/// - Note: This layer returns a tensor with shape `[batchSize, sequenceLength, hiddenSize]`.
///
/// - Source: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
/// Creates a transformer encoder.
///
/// - Parameters:
///   - hiddenSize: Size/depth of the transformer hidden representation.
///   - layerCount: Number of transformer layers.
///   - attentionHeadCount: Number of attention heads.
///   - attentionQueryActivation= Activation function applied to the attention query tensor.
///   - attentionKeyActivation= Activation function applied to the attention key tensor.
///   - attentionValueActivation= Activation function applied to the attention value tensor.
///   - intermediateSize: Size/depth of the transformer intermediate representation.
///   - intermediateActivation= Activation function applied to the intermediate representation.
///   - hiddenDropoutProbability: Dropout probability for the hidden representations.
///   - attentionDropoutProbability: Dropout probability for the attention scores.
///   - queryWeightInitializer: Initializer for the query transformation weight.
///   - queryBiasInitializer: Initializer for the query transformation bias.
///   - keyWeightInitializer: Initializer for the key transformation weight.
///   - keyBiasInitializer: Initializer for the key transformation bias.
///   - valueWeightInitializer: Initializer for the value transformation weight.
///   - valueBiasInitializer: Initializer for the value transformation bias.
///   - attentionWeightInitializer: Initializer for the attention transformation weight.
///   - attentionBiasInitializer: Initializer for the attention transformation bias.
///   - intermediateWeightInitializer: Initializer for the intermediate transformation weight.
///   - intermediateBiasInitializer: Initializer for the intermediate transformation bias.
///   - outputWeightInitializer: Initializer for the output transformation weight.
///   - outputBiasInitializer: Initializer for the output transformation bias.
type TransformerEncoder(hiddenSize: int,
        layerCount: int,
        attentionHeadCount: int,
        attentionQueryActivation: Activation,
        attentionKeyActivation: Activation,
        attentionValueActivation: Activation,
        intermediateSize: int,
        intermediateActivation: Activation,
        hiddenDropoutProbability: scalar,
        attentionDropoutProbability: scalar,
        ?queryWeightInitializer: ParameterInitializer,
        ?queryBiasInitializer: ParameterInitializer,
        ?keyWeightInitializer: ParameterInitializer,
        ?keyBiasInitializer: ParameterInitializer,
        ?valueWeightInitializer: ParameterInitializer,
        ?valueBiasInitializer: ParameterInitializer,
        ?attentionWeightInitializer: ParameterInitializer,
        ?attentionBiasInitializer: ParameterInitializer,
        ?intermediateWeightInitializer: ParameterInitializer,
        ?intermediateBiasInitializer: ParameterInitializer,
        ?outputWeightInitializer: ParameterInitializer,
        ?outputBiasInitializer: ParameterInitializer) = 
    inherit Model()

    let encoderLayers = 
        [| for _ in 0..layerCount-1 ->
            TransformerEncoderLayer(
                hiddenSize=hiddenSize,
                attentionHeadCount=attentionHeadCount,
                attentionQueryActivation= attentionQueryActivation,
                attentionKeyActivation= attentionKeyActivation,
                attentionValueActivation= attentionValueActivation,
                intermediateSize=intermediateSize,
                intermediateActivation= intermediateActivation,
                hiddenDropoutProbability=hiddenDropoutProbability,
                attentionDropoutProbability=attentionDropoutProbability,
                ?queryWeightInitializer=queryWeightInitializer,
                ?queryBiasInitializer=queryBiasInitializer,
                ?keyWeightInitializer=keyWeightInitializer,
                ?keyBiasInitializer=keyBiasInitializer,
                ?valueWeightInitializer=valueWeightInitializer,
                ?valueBiasInitializer=valueBiasInitializer,
                ?attentionWeightInitializer=attentionWeightInitializer,
                ?attentionBiasInitializer=attentionBiasInitializer,
                ?intermediateWeightInitializer=intermediateWeightInitializer,
                ?intermediateBiasInitializer=intermediateBiasInitializer,
                ?outputWeightInitializer=outputWeightInitializer,
                ?outputBiasInitializer=outputBiasInitializer) |]

    let regularizationValue =
        TangentVector
            {| encoderLayers = TangentVector(encoderLayers |> Array.map (fun x -> x.regularizationValue)) |}


    override _.forward(input: Tensor) : Tensor =
        let input = Unchecked.defaultof<TransformerInput> // TBD
        // The transformer performs sum residuals on all layers and so the input needs to have the
        // same depth as hidden size of the transformer.
        Debug.Assert(
            input.sequence.shape.[2] = hiddenSize,
            "The depth of the input tensor (\(input.sequence.shape.[2]) is different "
                + "than the hidden size ({hiddenSize}.")

        // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a
        // 3-D tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free
        // on TPUs, and so we want to minimize them to help the optimizer.
        let mutable transformerInput = input.sequence.reshapedToMatrix()
        let batchSize = input.sequence.shape.[0]
        for layerIndex in 0..encoderLayers.Length do
            let layerInput = 
                TransformerInput(sequence=transformerInput,
                    attentionMask=input.attentionMask,
                    batchSize=batchSize)
            transformerInput <- encoderLayers.[layerIndex].forward(failwith "tbd"(* layerInput *) )

        transformerInput.reshapedFromMatrix(input.sequence.shapex)

