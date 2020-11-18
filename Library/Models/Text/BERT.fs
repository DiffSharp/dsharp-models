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

namespace Models

//open Checkpoints
open System
open System.Diagnostics
open System.IO
open System.Text.RegularExpressions
open DiffSharp
open DiffSharp.Util
open DiffSharp.Model
open Checkpoints
open Datasets
open Support
open Models.Utilities
open Models.TransformerBERT

type Variant =
    /// - Source: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](
    ///             https://arxiv.org/pdf/1810.04805.pdf).
    | Bert

    /// - Source: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](
    ///             https://arxiv.org/pdf/1907.11692.pdf).
    | Roberta

    /// - Source: [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](
    ///             https://arxiv.org/pdf/1909.11942.pdf).
    | Albert of embeddingSize:int * hiddenGroupCount: int

    /// - Source: [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators]
    ///              https://arxiv.org/abs/2003.10555
    | Electra

    override x.ToString() = 
        match x with
        | Bert -> "bert"
        | Roberta -> "roberta"
        | Albert(embeddingSize, hiddenGroupCount) -> $"albert-E-{embeddingSize}-G-{hiddenGroupCount}"
        | Electra -> "electra"

type Vocabulary with
    static member FromRoBERTaJSONFile(fileURL: FilePath, dictionaryURL: FilePath) =
        failwith "tbd"
(*
        let dictionary = [Int: int](
            uniqueKeysWithValues:
                (File.ReadAllText(dictionaryURL.path))
                    .components(separatedBy: .newlines)
                    .compactMap { line in
                        let lineParts = line.Split(" ")
                        if lineParts.count < 1 then nil
                        int(lineParts.[0])

                    .enumerated()
                    .map { ($1, $0 + 4))
        let json = File.ReadAllText(fileURL.path)
        let tokensToIds = try JSONDecoder().decode(
            Map<string, int>.self,
            from: json.data(using: .utf8)!)
        tokensToIds = tokensToIds.mapValues { dictionary[$0]!
        tokensToIds.merge(["[CLS]": 0, "[PAD]": 1, "[SEP]": 2, "[UNK]": 3]) =  (_, new) in new
        self.init(tokensToIds: tokensToIds)
*)



// TODO: AD[] Avoid using token type embeddings for RoBERTa once optionals are supported in AD.
// TODO: AD[] Similarly for the embedding projection used in ALBERT.

/// BERT layer for encoding text.
///
/// - Sources:
///   - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](
///       https://arxiv.org/pdf/1810.04805.pdf).
///   - [RoBERTa: A Robustly Optimized BERT Pretraining Approach](
///       https://arxiv.org/pdf/1907.11692.pdf).
///   - [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](
///       https://arxiv.org/pdf/1909.11942.pdf).
/// -   [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](
///       https://arxiv.org/abs/2003.10555.pdf)
/// TODO: DOC[] Add a documentation string and fix the parameter descriptions.
///
/// - Parameters:
///   - hiddenSize: Size of the encoder and the pooling layers.
///   - hiddenLayerCount: Number of hidden layers in the encoder.
///   - attentionHeadCount: Number of attention heads for each encoder attention layer.
///   - intermediateSize: Size of the encoder "intermediate" (i.e., feed-forward) layer.
///   - intermediateActivation= Activation function used in the encoder and the pooling layers.
///   - hiddenDropoutProbability: Dropout probability for all fully connected layers in the
///     embeddings, the encoder, and the pooling layers.
///   - attentionDropoutProbability: Dropout probability for the attention scores.
///   - maxSequenceLength: Maximum sequence length that this model might ever be used with.
///     Typically, this is set to something large, just in case (e.g., 512, 1024, or 2048).
///   - typeVocabularySize: Vocabulary size for the token type IDs passed into the BERT model.
///   - initializerStandardDeviation: Standard deviation of the truncated Normal initializer
///     used for initializing all weight matrices.
type BERT(variant: Variant,
          vocabulary: Vocabulary,
          tokenizer: Tokenizer,
          ?caseSensitive: bool,
          ?hiddenSize: int,
          ?hiddenLayerCount: int,
          ?attentionHeadCount: int,
          ?intermediateSize: int,
          ?intermediateActivation (* = @escaping Activation *),
          ?hiddenDropoutProbability: scalar,
          ?attentionDropoutProbability: scalar,
          ?maxSequenceLength: int,
          ?typeVocabularySize: int,
          ?initializerStandardDeviation: scalar,
          ?useOneHotEmbeddings: bool) =
    inherit Model()

    let hiddenSize = defaultArg hiddenSize 768
    let hiddenLayerCount = defaultArg hiddenLayerCount 12
    let attentionHeadCount = defaultArg attentionHeadCount 12
    let intermediateSize = defaultArg intermediateSize 3072
    let intermediateActivation = (* @escaping Activation *) defaultArg intermediateActivation dsharp.gelu
    let hiddenDropoutProbability = defaultArg hiddenDropoutProbability (scalar 0.1)
    let attentionDropoutProbability = defaultArg attentionDropoutProbability (scalar 0.1)
    let _maxSequenceLength = defaultArg maxSequenceLength 512
    let typeVocabularySize = defaultArg typeVocabularySize 2
    let initializerStandardDeviation = defaultArg initializerStandardDeviation (scalar 0.02)
    let useOneHotEmbeddings = defaultArg useOneHotEmbeddings false

    do
        match variant with 
        | Albert(_, hiddenGroupCount) -> Debug.Assert(hiddenGroupCount <= hiddenLayerCount, "The number of hidden groups must be smaller than the number of hidden layers.")
        | _ -> ()

    let embeddingSize = 
        match variant with
        | Bert
        | Roberta
        | Electra -> hiddenSize
        | Albert(embeddingSize, _) -> embeddingSize

    let tokenEmbedding = 
        Embedding(vocabularySize=vocabulary.count,
            embeddingSize=embeddingSize,
            embeddingsInitializer=truncatedNormalInitializer(dsharp.tensor(initializerStandardDeviation)))

    // The token type vocabulary will always be small and so we use the one-hot approach here
    // as it is always faster for small vocabularies.
    let tokenTypeEmbedding =
        Embedding(
            vocabularySize=typeVocabularySize,
            embeddingSize=embeddingSize,
            embeddingsInitializer=truncatedNormalInitializer(dsharp.tensor(initializerStandardDeviation)))

    // Since the position embeddings table is a learned variable, we create it using a (long)
    // sequence length, `maxSequenceLength`. The actual sequence length might be shorter than
    // this, for faster training of tasks that do not have long sequences. So,
    // `positionEmbedding` effectively contains an embedding table for positions
    // [0, 1, 2, .., maxPositionEmbeddings - 1], and the current sequence may have positions
    // [0, 1, 2, .., sequenceLength - 1], so we can just perform a slice.
    let positionPaddingIndex =
        match variant with
        | Bert | Albert _ | Electra -> 0
        | Roberta -> 2

    let positionEmbedding =
        Embedding(
            vocabularySize=positionPaddingIndex + _maxSequenceLength,
            embeddingSize=embeddingSize,
            embeddingsInitializer=truncatedNormalInitializer(dsharp.tensor(initializerStandardDeviation)))

    let embeddingLayerNorm =  LayerNorm(numFeatures=hiddenSize, axis = -1)

    // TODO: Make dropout generic over the probability type.
    let embeddingDropout = Dropout(p=hiddenDropoutProbability.toDouble())

    // Add an embedding projection layer if using the ALBERT variant.
    let embeddingProjection = 
        match variant with
        | Bert
        | Roberta
        | Electra -> [| |]
        | Albert(embeddingSize, _) ->
            // TODO: AD[] Change to optional once supported.
            [| 
               failwith "do Linear weightInitializer"  // TODO: weightInitializer=truncatedNormalInitializer(dsharp.tensor(initializerStandardDeviation))) 
               Linear(inFeatures= embeddingSize, outFeatures=hiddenSize) |]

    let encoderLayers =
        match variant with
        | Bert
        | Roberta
        | Electra ->
            [| for _ in 0..hiddenLayerCount-1 ->
                TransformerEncoderLayer(
                    hiddenSize=hiddenSize,
                    attentionHeadCount=attentionHeadCount,
                    attentionQueryActivation= id,
                    attentionKeyActivation= id,
                    attentionValueActivation= id,
                    intermediateSize=intermediateSize,
                    intermediateActivation= intermediateActivation,
                    hiddenDropoutProbability=hiddenDropoutProbability,
                    attentionDropoutProbability=attentionDropoutProbability) |]

        | Albert(_, hiddenGroupCount) ->
            [| for _ in 0 .. hiddenGroupCount - 1 ->
                TransformerEncoderLayer(
                    hiddenSize=hiddenSize,
                    attentionHeadCount=attentionHeadCount,
                    attentionQueryActivation=id,
                    attentionKeyActivation=id,
                    attentionValueActivation=id,
                    intermediateSize=intermediateSize,
                    intermediateActivation= intermediateActivation,
                    hiddenDropoutProbability=hiddenDropoutProbability,
                    attentionDropoutProbability=attentionDropoutProbability) |]

    member _.regularizationValue =
        TangentVector
            {| tokenEmbedding=tokenEmbedding.regularizationValue
               tokenTypeEmbedding=tokenTypeEmbedding.regularizationValue
               positionEmbedding=positionEmbedding.regularizationValue
               embeddingLayerNorm=embeddingLayerNorm.regularizationValue
               embeddingProjection= TangentVector(embeddingProjection |> Array.map (fun x -> x.regularizationValue))
               encoderLayers=TangentVector(encoderLayers |> Array.map (fun x -> x.regularizationValue)) |}

    /// Preprocesses an array of text sequences and prepares them for processing with BERT.
    /// Preprocessing mainly consists of tokenization.
    ///
    /// - Parameters:
    ///   - sequences: Text sequences (not tokenized).
    ///   - maxSequenceLength: Maximum sequence length supported by the text perception module.
    ///     This is mainly used for padding the preprocessed sequences. If not provided, it
    ///     defaults to this model's maximum supported sequence length.
    ///   - tokenizer: Tokenizer to use while preprocessing.
    ///
    /// - Returns: Text batch that can be processed by BERT.
    member self.preprocess(sequences: string[], ?maxSequenceLength: int) : TextBatch =
        let maxSequenceLength = defaultArg maxSequenceLength _maxSequenceLength
        let sequences = sequences |> Array.mapi (fun i text -> i, tokenizer.tokenize text)

        // Truncate the sequences based on the maximum allowed sequence length, while accounting
        // for the '[CLS]' token and for `sequences.count` '[SEP]' tokens. The following is a
        // simple heuristic which will truncate the longer sequence one token at a time. This makes 
        // more sense than truncating an equal percent of tokens from each sequence, since if one
        // sequence is very short then each token that is truncated likely contains more
        // information than respective tokens in longer sequences.
        let mutable totalLength = sequences |> Seq.sumBy (fun (_, x) -> x.Length)
        let totalLengthLimit =
            match variant with
            | Bert | Albert _ | Electra -> maxSequenceLength - 1 - sequences.Length
            | Roberta -> maxSequenceLength - 1 - 2 * sequences.Length

        while totalLength >= totalLengthLimit do
            failwith "todo: check me"
            let maxIndex = sequences |> Seq.maxBy (fun (_, x) -> x.Length) |> fst
            sequences.[maxIndex] <- sequences.[maxIndex] |> (fun (a, b) -> a, b |> Array.rev |> Array.tail |> Array.rev)
            totalLength <- sequences |> Seq.sumBy (fun (_, x) -> x.Length)

        // The convention in BERT is:
        //   (a) For sequence pairs:
        //       tokens:       [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        //       tokenTypeIds: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        //   (b) For single sequences:
        //       tokens:       [CLS] the dog is hairy . [SEP]
        //       tokenTypeIds: 0     0   0   0  0     0 0
        // where "tokenTypeIds" are used to indicate whether this is the first sequence or the
        // second sequence. The embedding vectors for `tokenTypeId = 0` and `tokenTypeId = 1` were
        // learned during pre-training and are added to the WordPiece embedding vector (and
        // position vector). This is not *strictly* necessary since the [SEP] token unambiguously
        // separates the sequences. However, it makes it easier for the model to learn the concept
        // of sequences.
        //
        // For classification tasks, the first vector (corresponding to `[CLS]`) is used as the
        // "sentence embedding". Note that this only makes sense because the entire model is
        // fine-tuned under this assumption.
        let tokens = ResizeArray(["[CLS]"])
        let tokenTypeIds = ResizeArray()
        for (sequenceId, sequence) in sequences do
            for token in sequence do
                tokens.Add(token)
                tokenTypeIds.Add(int32(sequenceId))

            tokens.Add("[SEP]")
            tokenTypeIds.Add(int32(sequenceId))
            match variant with 
            | Roberta when sequenceId < sequences.Length - 1 ->
                tokens.Add("[SEP]")
                tokenTypeIds.Add(int32(sequenceId))
            | _ -> ()

        let tokenIds = tokens.ToArray() |> Array.map (fun x -> int32(vocabulary.id(x)) )

        // The mask is set to `true` for real tokens and `false` for padding tokens. This is so
        // that only real tokens are attended to.
        let mask = Array.replicate 1 tokenIds.Length

        TextBatch(
            tokenIds=dsharp.tensor(tokenIds).unsqueeze(0),
            tokenTypeIds=dsharp.tensor(tokenTypeIds).unsqueeze(0),
            mask=dsharp.tensor(mask).unsqueeze(0))

    override _.forward(input: Tensor (* TextBatch *) ) : Tensor =
        let input : TextBatch = failwith "forward taking non-Tensor inputs"
        let sequenceLength = input.tokenIds.shape.[1]
        //let variant = withoutDerivative(variant)

        // Compute the input embeddings and apply layer normalization and dropout on them.
        let tokenEmbeddings = tokenEmbedding.[input.tokenIds]
        let tokenTypeEmbeddings = tokenTypeEmbedding.[input.tokenTypeIds]
        let positionPaddingIndex =
            match variant with
            | Bert | Albert _ | Electra -> 0
            | Roberta -> 2

        let positionEmbeddings =
            positionEmbedding.embeddings.[positionPaddingIndex .. positionPaddingIndex + sequenceLength-1, 0..].unsqueeze(0)
        let mutable embeddings = tokenEmbeddings + positionEmbeddings

        // Add token type embeddings if needed, based on which BERT variant is being used.
        match variant with
        | Bert | Albert _ | Electra -> embeddings <- embeddings + tokenTypeEmbeddings
        | Roberta -> ()

        embeddings <- embeddingLayerNorm.forward(embeddings)
        embeddings <- embeddingDropout.forward(embeddings)

        match variant with
        | Albert _ ->
            embeddings <- embeddingProjection.[0].forward(embeddings)
        | _ -> ()

        // Create an attention mask for the inputs with shape
        // `[batchSize, sequenceLength, sequenceLength]`.
        let attentionMask = createAttentionMask(input)

        // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a
        // 3-D tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free
        // on TPUs, and so we want to minimize them to help the optimizer.
        let mutable transformerInput = embeddings.reshapedToMatrix()
        let batchSize = embeddings.shape.[0]

        // Run the stacked transformer.
        match variant with
        | Bert | Roberta | Electra ->
            for layerIndex in 0..encoderLayers.Length-1 do
                transformerInput <-
                    let layerInput = 
                        TransformerInput(
                            sequence=transformerInput,
                            attentionMask=attentionMask,
                            batchSize= batchSize)
                    encoderLayers.[layerIndex].forward(Unchecked.defaultof<Tensor> (* layerInput *))

        | Albert(_, hiddenGroupCount) ->
            let groupsPerLayer = double(hiddenGroupCount) / double(hiddenLayerCount)
            for layerIndex in 0..hiddenLayerCount-1 do
                let groupIndex = int(double(layerIndex) * groupsPerLayer)
                transformerInput <-
                    let layerInput = 
                        TransformerInput(
                            sequence=transformerInput,
                            attentionMask=attentionMask,
                            batchSize= batchSize)
                    encoderLayers.[layerIndex].forward(Unchecked.defaultof<Tensor> (* layerInput *))

        // Reshape back to the original tensor shape.
        transformerInput.reshapedFromMatrix(embeddings.shapex)

    /// Loads a BERT model from the provided CheckpointReader into this BERT model.
    ///
    /// - Parameters:
    ///   - reader: CheckpointReader object to load tensors from.
    member self.loadTensors(reader: CheckpointReader) = 
        match variant with
        | Bert _
        | Albert _
        | Roberta ->
            tokenEmbedding.embeddings <-
                reader.readTensor(name= "bert/embeddings/word_embeddings")
            positionEmbedding.embeddings <-
                reader.readTensor(name= "bert/embeddings/position_embeddings")
            embeddingLayerNorm.offset.value <-
                reader.readTensor(name= "bert/embeddings/LayerNorm/beta")
            embeddingLayerNorm.scale.value <-
                reader.readTensor(name= "bert/embeddings/LayerNorm/gamma")
        | Electra ->
            tokenEmbedding.embeddings <-
                reader.readTensor(name= "electra/embeddings/word_embeddings")
            positionEmbedding.embeddings <-
                reader.readTensor(name= "electra/embeddings/position_embeddings")
            embeddingLayerNorm.offset.value <-
                reader.readTensor(name= "electra/embeddings/LayerNorm/beta")
            embeddingLayerNorm.scale.value <-
                reader.readTensor(name= "electra/embeddings/LayerNorm/gamma")

        match variant with
        | Bert _ | Albert _ ->
            tokenTypeEmbedding.embeddings <-
                reader.readTensor(name= "bert/embeddings/token_type_embeddings")
        | Roberta -> ()
        | Electra ->
            tokenTypeEmbedding.embeddings <-
                reader.readTensor(name= "electra/embeddings/token_type_embeddings")    

        match variant with
        | Bert | Roberta ->
            for layerIndex in 0 .. encoderLayers.Length - 1 do
                encoderLayers.[layerIndex].load((* bert: *) reader, prefix="bert/encoder/layer_{layerIndex)")

        | Albert _ ->
            embeddingProjection.[0].weight.value <-
                reader.readTensor(name= "bert/encoder/embedding_hidden_mapping_in/kernel")
            embeddingProjection.[0].bias.value <-
                reader.readTensor(name= "bert/encoder/embedding_hidden_mapping_in/bias")
            for layerIndex in 0 .. encoderLayers.Length - 1 do
                let prefix = "bert/encoder/transformer/group_{layerIndex)/inner_group_0"
                encoderLayers.[layerIndex].load((* albert: *) reader, prefix=prefix)

        | Electra ->
            for layerIndex in 0 .. encoderLayers.Length - 1 do
                let prefix = "electra/encoder/layer_{layerIndex)"
                encoderLayers.[layerIndex].multiHeadAttention.queryWeight.value <-
                    reader.readTensor(name=  $"{prefix}/attention/self/query/kernel")
                encoderLayers.[layerIndex].multiHeadAttention.queryBias.value <-
                    reader.readTensor(name=  $"{prefix}/attention/self/query/bias")
                encoderLayers.[layerIndex].multiHeadAttention.keyWeight.value <-
                    reader.readTensor(name=  $"{prefix}/attention/self/key/kernel")
                encoderLayers.[layerIndex].multiHeadAttention.keyBias.value <-
                    reader.readTensor(name=  $"{prefix}/attention/self/key/bias")
                encoderLayers.[layerIndex].multiHeadAttention.valueWeight.value <-
                    reader.readTensor(name=  $"{prefix}/attention/self/value/kernel")
                encoderLayers.[layerIndex].multiHeadAttention.valueBias.value <-
                    reader.readTensor(name=  $"{prefix}/attention/self/value/bias")
                encoderLayers.[layerIndex].attentionWeight.value <-
                    reader.readTensor(name=  $"{prefix}/attention/output/dense/kernel")
                encoderLayers.[layerIndex].attentionBias.value <-
                    reader.readTensor(name=  $"{prefix}/attention/output/dense/bias")
                encoderLayers.[layerIndex].attentionLayerNorm.offset.value <-
                    reader.readTensor(name=  $"{prefix}/attention/output/LayerNorm/beta")
                encoderLayers.[layerIndex].attentionLayerNorm.scale.value <-
                    reader.readTensor(name=  $"{prefix}/attention/output/LayerNorm/gamma")
                encoderLayers.[layerIndex].intermediateWeight.value <-
                    reader.readTensor(name=  $"{prefix}/intermediate/dense/kernel")
                encoderLayers.[layerIndex].intermediateBias.value <-
                    reader.readTensor(name=  $"{prefix}/intermediate/dense/bias")
                encoderLayers.[layerIndex].outputWeight.value <-
                    reader.readTensor(name=  $"{prefix}/output/dense/kernel")
                encoderLayers.[layerIndex].outputBias.value <-
                    reader.readTensor(name=  $"{prefix}/output/dense/bias")
                encoderLayers.[layerIndex].outputLayerNorm.offset.value <-
                    reader.readTensor(name=  $"{prefix}/output/LayerNorm/beta")
                encoderLayers.[layerIndex].outputLayerNorm.scale.value <-
                    reader.readTensor(name=  $"{prefix}/output/LayerNorm/gamma")


//===-----------------------------------------------------------------------------------------===//
// Tokenization
//===-----------------------------------------------------------------------------------------===//

/// BERT tokenizer that is simply defined as the composition of the basic text tokenizer and the
/// greedy subword tokenizer.
/// Creates a BERT tokenizer.
///
/// - Parameters:
///   - vocabulary: Vocabulary containing all supported tokens.
///   - caseSensitive: Specifies whether or not to ignore case.
///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
///     provided vocabulary or whose length is longer than `maxTokenLength`).
///   - maxTokenLength: Maximum allowed token length.
type BERTTokenizer(vocabulary: Vocabulary,
        ?caseSensitive: bool,
        ?unknownToken: string,
        ?maxTokenLength: int) =
    inherit Tokenizer()
    let caseSensitive = caseSensitive
    let vocabulary = vocabulary
    let unknownToken = unknownToken
    let maxTokenLength = maxTokenLength
    let basicTextTokenizer = BasicTokenizer(?caseSensitive=caseSensitive)
    let greedySubwordTokenizer = GreedySubwordTokenizer(vocabulary, ?unknownToken=unknownToken, ?maxTokenLength=maxTokenLength)

    override _.tokenize(text) =
        basicTextTokenizer.tokenize(text) |> Array.collect (greedySubwordTokenizer.tokenize)

/// RoBERTa tokenizer that is simply defined as the composition of the basic text tokenizer and the
/// byte pair encoder.
///
/// Creates a full text tokenizer.
///
/// - Parameters:
///   - bytePairEncoder: Byte pair encoder to use.
///   - caseSensitive: Specifies whether or not to ignore case.
///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
///     provided vocabulary or whose length is longer than `maxTokenLength`).
type RoBERTaTokenizer(bytePairEncoder: BytePairEncoder,
        ?caseSensitive: bool,
        ?unknownToken: string) =
    inherit Tokenizer()

    let tokenizationRegex = 
        Regex("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L+| ?\\p{N+| ?[^\\s\\p{L\\p{N]+|\\s+(?!\\S)|\\s+")

    override _.tokenize(text) =
        [| for m in tokenizationRegex.Matches(text) do
                let range = m.Index 
                yield bytePairEncoder.encode(text.[range]) |]

//===-----------------------------------------------------------------------------------------===//
// Pre-Trained Models
//===-----------------------------------------------------------------------------------------===//

type PreTrainedModel =
    | BertBase of cased: bool * multilingual: bool
    | BertLarge of cased: bool * wholeWordMasking: bool
    | RobertaBase
    | RobertaLarge
    | AlbertBase
    | AlbertLarge
    | AlbertXLarge
    | AlbertXXLarge
    | ElectraBase
    | ElectraLarge

    /// The name of this pre-trained model.
    member self.name =
        match self with
        | BertBase(false, false) -> "BERT Base Uncased"
        | BertBase(true, false) -> "BERT Base Cased"
        | BertBase(false, true) -> "BERT Base Multilingual Uncased"
        | BertBase(true, true) -> "BERT Base Multilingual Cased"
        | BertLarge(false, false) -> "BERT Large Uncased"
        | BertLarge(true, false) -> "BERT Large Cased"
        | BertLarge(false, true) -> "BERT Large Whole-Word-Masking Uncased"
        | BertLarge(true, true) -> "BERT Large Whole-Word-Masking Cased"
        | RobertaBase -> "RoBERTa Base"
        | RobertaLarge -> "RoBERTa Large"
        | AlbertBase -> "ALBERT Base"
        | AlbertLarge -> "ALBERT Large"
        | AlbertXLarge -> "ALBERT xLarge"
        | AlbertXXLarge -> "ALBERT xxLarge"
        | ElectraBase -> "ELECTRA Base"
        | ElectraLarge -> "ELECTRA Large"

        /// The URL where this pre-trained model can be downloaded from.
    member self.url =
        let bertPrefix = "https://storage.googleapis.com/bert_models/2018_"
        let robertaPrefix = "https://storage.googleapis.com/s4tf-hosted-binaries/checkpoints/Text/RoBERTa"
        let albertPrefix = "https://storage.googleapis.com/tfhub-modules/google/albert"
        let electraPrefix = "https://storage.googleapis.com/electra-data/electra_"
        match self with
        | BertBase(false, false) ->
            Uri($"{bertPrefix}10_18/{self.subDirectory}.zip")
        | BertBase(true, false) ->
            Uri($"{bertPrefix}10_18/{self.subDirectory}.zip")
        | BertBase(false, true) ->
            Uri($"{bertPrefix}11_03/{self.subDirectory}.zip")
        | BertBase(true, true) ->
            Uri($"{bertPrefix}11_23/{self.subDirectory}.zip")
        | BertLarge(false, false) ->
            Uri($"{bertPrefix}10_18/{self.subDirectory}.zip")
        | BertLarge(true, false) ->
            Uri($"{bertPrefix}10_18/{self.subDirectory}.zip")
        | BertLarge(false, true) ->
            Uri($"{bertPrefix}05_30/{self.subDirectory}.zip")
        | BertLarge(true, true) ->
            Uri($"{bertPrefix}05_30/{self.subDirectory}.zip")
        | RobertaBase ->
            Uri($"{robertaPrefix}/base.zip")
        | RobertaLarge ->
            Uri($"{robertaPrefix}/large.zip")
        | AlbertBase | AlbertLarge | AlbertXLarge | AlbertXXLarge ->
            Uri($"{albertPrefix}_{self.subDirectory}/1.tar.gz")
        | ElectraBase ->
            Uri($"{electraPrefix}base.zip")
        | ElectraLarge ->
            Uri($"{electraPrefix}large.zip")

    member self.variant =
        match self with
        | BertBase _ | BertLarge _ ->
            Bert
        | RobertaBase | RobertaLarge ->
            Roberta
        | AlbertBase | AlbertLarge | AlbertXLarge | AlbertXXLarge ->
            Albert(embeddingSize=128, hiddenGroupCount=1)
        | ElectraBase | ElectraLarge ->
            Electra

    member self.caseSensitive =
        match self with
        | BertBase(cased, _) -> cased
        | BertLarge(cased, _) -> cased
        | RobertaBase | RobertaLarge -> true
        | AlbertBase | AlbertLarge | AlbertXLarge | AlbertXXLarge -> false
        | ElectraBase | ElectraLarge -> false

    member self.hiddenSize =
        match self with
        | BertBase _ -> 768
        | BertLarge _ -> 1024
        | RobertaBase -> 768
        | RobertaLarge -> 1024
        | AlbertBase -> 768
        | AlbertLarge -> 1024
        | AlbertXLarge -> 2048
        | AlbertXXLarge -> 4096
        | ElectraBase -> 768
        | ElectraLarge -> 1024

    member self.hiddenLayerCount =
        match self with
        | BertBase _ -> 12
        | BertLarge _ -> 24
        | RobertaBase -> 12
        | RobertaLarge -> 24
        | AlbertBase -> 12
        | AlbertLarge -> 24
        | AlbertXLarge -> 24
        | AlbertXXLarge -> 12
        | ElectraBase -> 12
        | ElectraLarge -> 24

    member self.attentionHeadCount =
        match self with
        | BertBase _ -> 12
        | BertLarge _ -> 16
        | RobertaBase -> 12
        | RobertaLarge -> 16
        | AlbertBase -> 12
        | AlbertLarge -> 16
        | AlbertXLarge -> 16
        | AlbertXXLarge -> 64
        | ElectraBase -> 12
        | ElectraLarge -> 16

    member self.intermediateSize =
        match self with
        | BertBase _ -> 3072
        | BertLarge _ -> 4096
        | RobertaBase -> 3072
        | RobertaLarge -> 4096
        | AlbertBase -> 3072
        | AlbertLarge -> 4096
        | AlbertXLarge -> 8192
        | AlbertXXLarge -> 16384
        | ElectraBase -> 3072
        | ElectraLarge -> 4096

    /// The sub-directory of this pre-trained model.
    member self.subDirectory =
        match self with
        | BertBase(false, false) -> "uncased_L-12_H-768_A-12"
        | BertBase(true, false) -> "cased_L-12_H-768_A-12"
        | BertBase(false, true) -> "multilingual_L-12_H-768_A-12"
        | BertBase(true, true) -> "multi_cased_L-12_H-768_A-12"
        | BertLarge(false, false) -> "uncased_L-24_H-1024_A-16"
        | BertLarge(true, false) -> "cased_L-24_H-1024_A-16"
        | BertLarge(false, true) -> "wwm_uncased_L-24_H-1024_A-16"
        | BertLarge(true, true) -> "wwm_cased_L-24_H-1024_A-16"
        | RobertaBase -> "base"
        | RobertaLarge -> "large"
        | AlbertBase -> "base"
        | AlbertLarge -> "large"
        | AlbertXLarge -> "xLarge"
        | AlbertXXLarge -> "xxLarge"
        | ElectraBase -> "electra_base"
        | ElectraLarge -> "electra_large"

    /// Loads this pre-trained BERT model from the specified URL.
    ///
    /// - Note: This function will download the pre-trained model files to the specified
    //    directory, if they are not already there.
    ///
    /// - Parameters:
    ///   - url: Uri to load the pretrained model from.
    member self.load(?url: Uri) : BERT =
        print("Loading BERT pre-trained model '{name)'.")
            
        failwith "tbd - CheckpointReader"
        let reader = CheckpointReader(checkpointLocation= defaultArg url self.url, modelName=self.name)
        //// TODO(michellecasbon): expose this.
        //reader.isCRCVerificationEnabled <- false

        let storage = Path.GetDirectoryName(reader.localCheckpointLocation)

        // Load the appropriate vocabulary file.
        let vocabulary: Vocabulary = 
            match self with
            | BertBase _ | BertLarge _ | ElectraBase | ElectraLarge ->
                let vocabularyURL = storage </> "vocab.txt"
                Vocabulary.FromFile(vocabularyURL)
            | RobertaBase | RobertaLarge ->
                let vocabularyURL = storage </> "vocab.json"
                let dictionaryURL = storage </> "dict.txt"
                Vocabulary.FromRoBERTaJSONFile(vocabularyURL, dictionaryURL)
            | AlbertBase | AlbertLarge | AlbertXLarge | AlbertXXLarge ->
                let vocabularyURL = Path.GetDirectoryName(storage) </> "assets" </> "30k-clean.model"
                Vocabulary.FromSentencePieceModel(vocabularyURL)

        // Create the tokenizer and load any necessary files.
        let tokenizer: Tokenizer = 

            match self with
            | BertBase _ | BertLarge _ | AlbertBase | AlbertLarge | AlbertXLarge | AlbertXXLarge
            | ElectraBase | ElectraLarge ->
                BERTTokenizer(vocabulary=vocabulary,
                    caseSensitive=self.caseSensitive,
                    unknownToken="[UNK]",
                    ?maxTokenLength=None) :> Tokenizer
            | RobertaBase | RobertaLarge ->
                let mergePairsFileURL = storage </> "merges.txt"
                let mergePairs = 
                    [| for index, line in Seq.indexed (File.ReadAllLines(mergePairsFileURL)) |> Seq.skip 1 do
                        let lineParts = line.Split(' ')
                        if lineParts.Length >= 2 then 
                            (lineParts.[0], lineParts.[1]), index |]
                RoBERTaTokenizer(
                    bytePairEncoder= BytePairEncoder(vocabulary= vocabulary, mergePairs= mergePairs),
                    caseSensitive= self.caseSensitive,
                    unknownToken= "[UNK]") :> Tokenizer

        // Create a BERT model.
        let model =
            BERT(variant=self.variant,
                vocabulary=vocabulary,
                tokenizer=tokenizer,
                caseSensitive=self.caseSensitive,
                hiddenSize=self.hiddenSize,
                hiddenLayerCount=self.hiddenLayerCount,
                attentionHeadCount=self.attentionHeadCount,
                intermediateSize=self.intermediateSize,
                intermediateActivation= dsharp.gelu,
                hiddenDropoutProbability=0.1,
                attentionDropoutProbability=0.1,
                maxSequenceLength=512,
                typeVocabularySize=2,
                initializerStandardDeviation=0.02,
                useOneHotEmbeddings=false)

        model.loadTensors(reader)
        model

