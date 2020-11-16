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

open DiffSharp

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
type TransformerEncoderLayer(hiddenSize: int,
        attentionHeadCount: int,
        attentionQueryactivation: Activation<Scalar>,
        attentionKeyactivation: Activation<Scalar>,
        attentionValueactivation: Activation<Scalar>,
        intermediateSize: int,
        intermediateactivation: Activation<Scalar>,
        hiddenDropoutProbability: Scalar,
        attentionDropoutProbability: Scalar,
        queryWeightInitializer: ParameterInitializer<Scalar>,
        queryBiasInitializer: ParameterInitializer<Scalar>,
        keyWeightInitializer: ParameterInitializer<Scalar>,
        keyBiasInitializer: ParameterInitializer<Scalar>,
        valueWeightInitializer: ParameterInitializer<Scalar>,
        valueBiasInitializer: ParameterInitializer<Scalar>,
        attentionWeightInitializer: ParameterInitializer<Scalar>,
        attentionBiasInitializer: ParameterInitializer<Scalar>,
        intermediateWeightInitializer: ParameterInitializer<Scalar>,
        intermediateBiasInitializer: ParameterInitializer<Scalar>,
        outputWeightInitializer: ParameterInitializer<Scalar>,
        outputBiasInitializer: ParameterInitializer<Scalar>) = 
    do 
        Debug.Assert(
            hiddenSize % attentionHeadCount = 0,
            "The hidden size of the transformer ({hiddenSize}) must be a multiple of the "
                + "attention head count ({attentionHeadCount}).")
    self.hiddenSize = hiddenSize
    self.intermediateActivation = intermediateActivation
    self.multiHeadAttention = MultiHeadAttention(
        sourceSize: hiddenSize,
        targetSize: hiddenSize,
        headCount: attentionHeadCount,
        headSize: hiddenSize / attentionHeadCount,
        queryactivation= attentionQueryActivation,
        keyactivation= attentionKeyActivation,
        valueactivation= attentionValueActivation,
        attentionDropoutProbability: attentionDropoutProbability,
        matrixResult: true,
        queryWeightInitializer: queryWeightInitializer,
        queryBiasInitializer: queryBiasInitializer,
        keyWeightInitializer: keyWeightInitializer,
        keyBiasInitializer: keyBiasInitializer,
        valueWeightInitializer: valueWeightInitializer,
        valueBiasInitializer: valueBiasInitializer)
    // TODO: Make dropout generic over the probability type.
    self.hiddenDropout = Dropout(probability: Double(hiddenDropoutProbability))
    self.attentionWeight = attentionWeightInitializer(
        [attentionHeadCount * hiddenSize / attentionHeadCount, hiddenSize])
    self.attentionBias = attentionBiasInitializer([hiddenSize])
    self.attentionLayerNorm = LayerNorm( featureCount=hiddenSize, axis=-1)
    self.intermediateWeight = intermediateWeightInitializer([hiddenSize, intermediateSize])
    self.intermediateBias = intermediateBiasInitializer([intermediateSize])
    self.outputWeight = intermediateWeightInitializer([intermediateSize, hiddenSize])
    self.outputBias = intermediateBiasInitializer([hiddenSize])
    self.outputLayerNorm = LayerNorm(featureCount=hiddenSize, axis = -1)

    let regularizationValue: TangentVector {
        TangentVector(
            multiHeadAttention: multiHeadAttention.regularizationValue,
            attentionWeight: attentionWeight,
            attentionBias: dsharp.tensor(Scalar(0), device=attentionBias.device),
            attentionLayerNorm: attentionLayerNorm.regularizationValue,
            intermediateWeight: intermediateWeight,
            intermediateBias: dsharp.tensor(Scalar(0), device=intermediateBias.device),
            outputWeight: outputWeight,
            outputBias: dsharp.tensor(Scalar(0), device=outputBias.device),
            outputLayerNorm: outputLayerNorm.regularizationValue)


    /// Creates a transformer encoder layer.
    ///
    /// - Parameters:
    ///   - hiddenSize: Size/depth of the transformer hidden representation.
    ///   - attentionHeadCount: Number of attention heads.
    ///   - attentionQueryactivation= Activation function applied to the attention query tensor.
    ///   - attentionKeyactivation= Activation function applied to the attention key tensor.
    ///   - attentionValueactivation= Activation function applied to the attention value tensor.
    ///   - intermediateSize: Size/depth of the transformer intermediate representation.
    ///   - intermediateactivation= Activation function applied to the intermediate representation.
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
    
    override _.forward(input: TransformerInput<Scalar>) : Tensor =
        let attentionInput = AttentionInput(
            source: input.sequence,
            target: input.sequence,
            mask: input.attentionMask,
            batchSize= input.batchSize)
        let attentionOutput = multiHeadAttention(attentionInput)

        // Run a linear projection of `hiddenSize` and then add a residual connection to the input.
        attentionOutput = matmul(attentionOutput, attentionWeight) + attentionBias
        attentionOutput = hiddenDropout(attentionOutput)
        attentionOutput = attentionLayerNorm(attentionOutput + input.sequence)

        // The activation is only applied to the "intermediate" hidden layer.
        let intermediateOutput = matmul(attentionOutput, intermediateWeight) + intermediateBias
        intermediateOutput = intermediateActivation(intermediateOutput)

        // Project back to `hiddenSize` and add the residual.
        let output = matmul(intermediateOutput, outputWeight) + outputBias
        output = hiddenDropout(output)
        output = outputLayerNorm(output + attentionOutput)

        return output



extension TransformerEncoderLayer {
    /// Default initializer to use for the linear transform weights.
    public static let defaultWeightInitializer: ParameterInitializer<Scalar> {
        truncatedNormalInitializer(standardDeviation=Tensor(0.02))


    /// Default initializer to use for the linear transform biases.
    public static let defaultBiasInitializer: ParameterInitializer<Scalar> {
        zeros()


/// Multi-headed and multi-layer transformer encoder.
///
/// - Note: This layer returns a tensor with shape `[batchSize, sequenceLength, hiddenSize]`.
///
/// - Source: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
type TransformerEncoder(hiddenSize: int,
        layerCount: int,
        attentionHeadCount: int,
        attentionQueryactivation: Activation<Scalar>,
        attentionKeyactivation: Activation<Scalar>,
        attentionValueactivation: Activation<Scalar>,
        intermediateSize: int,
        intermediateactivation: Activation<Scalar>,
        hiddenDropoutProbability: Scalar,
        attentionDropoutProbability: Scalar,
        ?queryWeightInitializer: ParameterInitializer<Scalar>,
        ?queryBiasInitializer: ParameterInitializer<Scalar>,
        ?keyWeightInitializer: ParameterInitializer<Scalar>,
        ?keyBiasInitializer: ParameterInitializer<Scalar>,
        ?valueWeightInitializer: ParameterInitializer<Scalar>,
        ?valueBiasInitializer: ParameterInitializer<Scalar>,
        ?attentionWeightInitializer: ParameterInitializer<Scalar>,
        ?attentionBiasInitializer: ParameterInitializer<Scalar>,
        ?intermediateWeightInitializer: ParameterInitializer<Scalar>,
        ?intermediateBiasInitializer: ParameterInitializer<Scalar>,
        ?outputWeightInitializer: ParameterInitializer<Scalar>,
        ?outputBiasInitializer: ParameterInitializer<Scalar>) = 
    inherit Model()
    let queryWeightInitializer = defaultArg defaultWeightInitializer,
    let queryBiasInitializer = defaultArg  defaultBiasInitializer,
    let keyWeightInitializer = defaultArg  defaultWeightInitializer,
    let keyBiasInitializer = defaultArg  defaultBiasInitializer,
    let valueWeightInitializer = defaultArg  defaultWeightInitializer,
    let valueBiasInitializer = defaultArg  defaultBiasInitializer,
    let attentionWeightInitializer = defaultArg  defaultWeightInitializer,
    let attentionBiasInitializer = defaultArg  defaultBiasInitializer,
    let intermediateWeightInitializer = defaultArg  defaultWeightInitializer,
    let intermediateBiasInitializer = defaultArg  defaultBiasInitializer,
    let outputWeightInitializer = defaultArg  defaultWeightInitializer,
    let outputBiasInitializer = defaultArg  defaultBiasInitializer) = 

    let encoderLayers = 
        [| for _ in 0..layerCount-1 ->
            TransformerEncoderLayer(
                hiddenSize: hiddenSize,
                attentionHeadCount: attentionHeadCount,
                attentionQueryactivation= attentionQueryActivation,
                attentionKeyactivation= attentionKeyActivation,
                attentionValueactivation= attentionValueActivation,
                intermediateSize: intermediateSize,
                intermediateactivation= intermediateActivation,
                hiddenDropoutProbability: hiddenDropoutProbability,
                attentionDropoutProbability: attentionDropoutProbability,
                queryWeightInitializer: queryWeightInitializer,
                queryBiasInitializer: queryBiasInitializer,
                keyWeightInitializer: keyWeightInitializer,
                keyBiasInitializer: keyBiasInitializer,
                valueWeightInitializer: valueWeightInitializer,
                valueBiasInitializer: valueBiasInitializer,
                attentionWeightInitializer: attentionWeightInitializer,
                attentionBiasInitializer: attentionBiasInitializer,
                intermediateWeightInitializer: intermediateWeightInitializer,
                intermediateBiasInitializer: intermediateBiasInitializer,
                outputWeightInitializer: outputWeightInitializer,
                outputBiasInitializer: outputBiasInitializer) |]

    // TODO: Convert to a generic constraint once TF-427 is resolved.
    type Scalar = Float

    let hiddenSize: int

    let encoderLayers: TransformerEncoderLayer[]

    let regularizationValue: TangentVector {
        TangentVector(
            encoderLayers: TransformerEncoderLayer[].TangentVector(
                encoderLayers.map (fun x -> x.regularizationValue))


    /// Creates a transformer encoder.
    ///
    /// - Parameters:
    ///   - hiddenSize: Size/depth of the transformer hidden representation.
    ///   - layerCount: Number of transformer layers.
    ///   - attentionHeadCount: Number of attention heads.
    ///   - attentionQueryactivation= Activation function applied to the attention query tensor.
    ///   - attentionKeyactivation= Activation function applied to the attention key tensor.
    ///   - attentionValueactivation= Activation function applied to the attention value tensor.
    ///   - intermediateSize: Size/depth of the transformer intermediate representation.
    ///   - intermediateactivation= Activation function applied to the intermediate representation.
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
    public init
    override _.forward(input: TransformerInput<Scalar>) : Tensor =
        // The transformer performs sum residuals on all layers and so the input needs to have the
        // same depth as hidden size of the transformer.
        Debug.Assert(
            input.sequence.shape.[2] = hiddenSize,
            "The depth of the input tensor (\(input.sequence.shape.[2]) is different "
                + "than the hidden size ({hiddenSize}.")

        // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a
        // 3-D tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free
        // on TPUs, and so we want to minimize them to help the optimizer.
        let transformerInput = input.sequence.reshapedToMatrix()
        let batchSize = input.sequence.shape.[0]
        for layerIndex in 0..<(withoutDerivative(at: encoderLayers) =  $0.count) = 
            transformerInput = encoderLayers.[layerIndex](
                TransformerInput(
                    sequence: transformerInput,
                    attentionMask: input.attentionMask,
                    batchSize= batchSize))


        return transformerInput.reshapedFromMatrix(originalShape: input.sequence.shape)



extension TransformerEncoder {
    /// Default initializer to use for the linear transform weights.
    public static let defaultWeightInitializer: ParameterInitializer<Scalar> {
        truncatedNormalInitializer(standardDeviation=Tensor(0.02))


    /// Default initializer to use for the linear transform biases.
    public static let defaultBiasInitializer: ParameterInitializer<Scalar> {
        zeros()




