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

module Models.Text.Attention

open System.Diagnostics
open DiffSharp
open DiffSharp.Model
open DiffSharp.ShapeChecking
open Models.Text.Utilities

[<AutoOpen>]
module Defaults =
    /// Default initializer to use for the linear transform weights.
    let defaultWeightInitializer = truncatedNormalInitializer(dsharp.scalar 0.02)

    /// Default initializer to use for the linear transform biases.
    let defaultBiasInitializer (shape: Shape) = dsharp.zeros shape


/// Input to an attention layer.
type AttentionInput(source: Tensor, target: Tensor, mask: Tensor, ?batchSize: int) =

    do Debug.Assert(source.ndims = target.ndims, "The rank of the attention source and target tensors must match.")

    /// Source tensor that we are attending from, with shape
    /// `[batchSize, sourceSequenceLength, sourceDepth]` or
    /// `[batchSize, sourceSequenceLength * sourceDepth]`.
    member _.source = source

    /// Target tensor that we are attending to, with shape
    /// `[batchSize, targetSequenceLength, targetDepth]` or
    /// `[batchSize, targetSequenceLength * targetDepth]`.
    member _.target = target

    /// Mask to apply on the attention scores. This is a tensor with shape
    /// `[batchSize, sourceSequenceLength, targetSequenceLength]` or
    /// `[batchSize, sourceSequenceLength * targetSequenceLength]`. The values should be `1` or `0`.
    /// The attention scores will effectively be set to negative infinity for any positions in the
    /// mask that are set to `0`, and will be unchanged for positions that are set to `1`.
    member _.mask = mask

    /// The batch size of this input. This is optional because it is only needed if the input
    /// sequences have been reshaped to matrices.
    member _.batchSize = batchSize

/// Multi-head attention layer.
///
/// This implementation is based on the
/// ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. If the source and target
/// tensors are the same, then this layer behaves as a self-attention layer. Each sequence step in
/// the source tensor attends to the corresponding sequence in the target tensor and returns a
/// fixed-size vector.
///
/// This function first projects the source tensor into a "query" tensor and the target tensor into
/// "key" and "value" tensors. These are (effectively) a list of tensors of length `headCount`,
/// where each tensor has shape `[batchSize, sequenceLength, headSize]`. It then performs a dot
/// product between the query and they key tensors and scales them. Finally, they are passed
/// through the softmax function to obtain attention probabilities. The value tensors are then
/// interpolated by these probabilities, and then concatenated back to a single result tensor.
///
/// In practice, the multi-head attention is implemented using transpose and reshape operations,
/// rather than using separate tensors.
///
/// - Source: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
/// Creates a multi-head attention layer.
///
/// - Parameters:
///   - sourceSize: Size/depth of the source tensor this layer is attending from.
///   - targetSize: Size/depth of the target tensor this layer is attending to.
///   - headCount: Number of attention heads.
///   - headSize: Size/depth of each attention head.
///   - queryActivation= Activation function applied to the attention query tensor.
///   - keyActivation= Activation function applied to the attention key tensor.
///   - valueActivation= Activation function applied to the attention value tensor.
///   - attentionDropoutProbability: Dropout probability for the attention scores.
///   - matrixResult: If `true`, the resulting tensor will have shape
///     `[batchSize * sourceSequenceLength, headCount * headSize]`. Otherwise, it will have shape
///     `[batchSize, sourceSequenceLength, headCount * headSize]`.
///   - queryWeightInitializer: Initializer for the query transformation weight.
///   - queryBiasInitializer: Initializer for the query transformation bias.
///   - keyWeightInitializer: Initializer for the key transformation weight.
///   - keyBiasInitializer: Initializer for the key transformation bias.
///   - valueWeightInitializer: Initializer for the value transformation weight.
///   - valueBiasInitializer: Initializer for the value transformation bias.
type MultiHeadAttention(sourceSize: int,
        targetSize: int,
        ?headCount: int,
        ?headSize: int,
        ?queryActivation: Activation,
        ?keyActivation: Activation,
        ?valueActivation: Activation,
        ?attentionDropoutProbability: scalar,
        ?matrixResult: bool,
        ?queryWeightInitializer: ParameterInitializer,
        ?queryBiasInitializer: ParameterInitializer,
        ?keyWeightInitializer: ParameterInitializer,
        ?keyBiasInitializer: ParameterInitializer,
        ?valueWeightInitializer: ParameterInitializer,
        ?valueBiasInitializer: ParameterInitializer) = 

    inherit Model<AttentionInput, Tensor>()

    let headCount = defaultArg headCount 1
    let headSize = defaultArg headSize 512
    let queryActivation = defaultArg queryActivation id
    let keyActivation = defaultArg keyActivation id
    let valueActivation = defaultArg valueActivation id
    let attentionDropoutProbability = defaultArg attentionDropoutProbability (scalar 0)
    let matrixResult = defaultArg matrixResult false
    let queryWeightInitializer = defaultArg queryWeightInitializer defaultWeightInitializer
    let queryBiasInitializer = defaultArg queryBiasInitializer defaultBiasInitializer
    let keyWeightInitializer = defaultArg keyWeightInitializer defaultWeightInitializer
    let keyBiasInitializer = defaultArg keyBiasInitializer defaultBiasInitializer
    let valueWeightInitializer = defaultArg valueWeightInitializer defaultWeightInitializer
    let valueBiasInitializer = defaultArg valueBiasInitializer defaultBiasInitializer

    let p_queryWeight = queryWeightInitializer(Shape [sourceSize; headCount * headSize]) |> Parameter
    let p_queryBias = queryBiasInitializer(Shape [headCount * headSize]) |> Parameter
    let p_keyWeight = keyWeightInitializer(Shape [targetSize; headCount * headSize]) |> Parameter
    let p_keyBias = keyBiasInitializer(Shape [headCount * headSize]) |> Parameter
    let p_valueWeight = valueWeightInitializer(Shape [targetSize; headCount * headSize]) |> Parameter
    let p_valueBias = valueBiasInitializer(Shape [headCount * headSize]) |> Parameter
    // TODO: Make dropout generic over the probability type.
    let attentionDropout = Dropout(attentionDropoutProbability.toDouble())

    member _.queryWeight = p_queryWeight
    member _.queryBias = p_queryBias
    member _.keyWeight = p_keyWeight
    member _.keyBias = p_keyBias
    member _.valueWeight = p_valueWeight
    member _.valueBias = p_valueBias

    member _.regularizationValue =
        TangentVector 
            {| queryWeight=p_queryWeight;
               queryBias=dsharp.tensor(0, device=p_queryBias.value.device);
               keyWeight=p_keyWeight;
               keyBias=dsharp.tensor(0, device=p_keyBias.value.device);
               valueWeight=p_valueWeight;
               valueBias=dsharp.tensor(0, device=p_valueBias.value.device) |}

    
    override _.forward(input: AttentionInput) : Tensor =
        Debug.Assert(
            input.source.ndims = 3 || input.batchSize.IsSome,
            "Whenever the input is provided in matrix form, the batch size must also be provided.")

        // Scalar dimensions referenced here:
        //   - B = batch size (number of sequences)
        //   - F = `input.source` sequence length
        //   - T = `input.target` sequence length
        //   - N = number of attention heads
        //   - H = size per attention head
        let matrixInput = input.source.ndims < 3
        let B = if matrixInput then input.batchSize.Value else input.source.shape.[0]
        let F = if matrixInput then input.source.shape.[0] / B else input.source.shape.[1]
        let T = if matrixInput then input.target.shape.[0] / B else input.target.shape.[1]
        let N = headCount
        let H = headSize

        let source = input.source.reshapedToMatrix()
        let target = input.target.reshapedToMatrix()

        let q = queryActivation(dsharp.matmul(source, p_queryWeight.value) + p_queryBias.value)  // [B * F; N * H]
        let k = keyActivation(dsharp.matmul(target, p_keyWeight.value) + p_keyBias.value)  // [B * T; N * H]
        let v = valueActivation(dsharp.matmul(target, p_valueWeight.value) + p_valueBias.value)  // [B * T; N * H]

        let q = q.view([B; F; N; H]).permute([| 0; 2; 1; 3 |])  // [B; N; F; H]
        let k = k.view([B; T; N; H]).permute([| 0; 2; 1; 3 |])  // [B; N; T; H]
        let v = v.view([B; T; N; H]).permute([| 0; 2; 1; 3 |])  // [B; N; T; H]

        // Take the dot product between the query and the key to get the raw attention scores.
        let attentionScores : Tensor = 
           failwith "todo - matmul transposed"
           dsharp.matmul(q, (* transposed: false, *) k (* ,  transposed: true *) )  // [B; N; F; T]
        let attentionScores = attentionScores / sqrt(double(headSize))

        // Since the attention mask is set to 1.0 for positions we want to attend to and 0.0 for
        // masked positions, we create a tensor which is 0.0 for positions we want to attend to and 
        // -10000.0 for masked positions. Since we are adding this tensor to the raw scores before 
        // the softmax, this is effectively the same as removing the masked entries entirely.
        let attentionMask = input.mask.unsqueeze(1)  // [B; 1, F; T]
        let attentionScores = attentionScores - 10000 * (1 - attentionMask)

        // Normalize the attention scores to convert them to probabilities. We are also dropping
        // out entire tokens to attend to, which might seem a bit unusual, but it is taken from the
        // original Transformer paper.
        let attentionProbabilities = attentionDropout.forward(dsharp.softmax(attentionScores, dim = -1))  // [B; N; F; T]

        let result = dsharp.matmul(attentionProbabilities, v)  // [B; N; F; H]
                           .permute([| 0; 2; 1; 3 |])  // [B; F; N; H]
        if matrixResult then result.view([B * F; N * H]) else result.view([B; F; N * H])


