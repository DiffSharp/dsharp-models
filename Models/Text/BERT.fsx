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

open Checkpoints
open Datasets


open DiffSharp

/// Represents a type that can contribute to the regularization term when training models.
type IRegularizable: Differentiable {
    /// The contribution of this term to the regularization term. This should be set to
    /// `TangentVector.zero` if this term should not contribute to the regularization term
    /// (e.g., for layer normalization parameters).
    let regularizationValue: TangentVector { get


extension Dense: Regularizable {
    let regularizationValue: TangentVector {
        TangentVector(weight: weight, bias: dsharp.tensor(Scalar(0), device=bias.device))



extension LayerNorm: Regularizable {
    let regularizationValue: TangentVector {
        TangentVector(
            offset: dsharp.tensor(Scalar(0), device=offset.device), scale: dsharp.tensor(Scalar(0), device=scale.device)
        )



extension Embedding: Regularizable {
    let regularizationValue: TangentVector {
        TangentVector(embeddings: embeddings)



// TODO: [AD] Avoid using token type embeddings for RoBERTa once optionals are supported in AD.
// TODO: [AD] Similarly for the embedding projection used in ALBERT.

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
type BERT: Module, Regularizable {
    // TODO: Convert to a generic constraint once TF-427 is resolved.
    type Scalar = Float

    let variant: Variant
    let vocabulary: Vocabulary
    let tokenizer: Tokenizer
    let caseSensitive: bool
    let hiddenSize: int
    let hiddenLayerCount: int
    let attentionHeadCount: int
    let intermediateSize: int
    let intermediateactivation= Activation<Scalar>
    let hiddenDropoutProbability: Scalar
    let attentionDropoutProbability: Scalar
    let maxSequenceLength: int
    let typeVocabularySize: int
    let initializerStandardDeviation: Scalar

    let tokenEmbedding: Embedding<Scalar>
    let tokenTypeEmbedding: Embedding<Scalar>
    let positionEmbedding: Embedding<Scalar>
    let embeddingLayerNorm: LayerNorm<Scalar>
    let embeddingDropout: Dropout<Scalar>
    let embeddingProjection: [Dense<Scalar>]
    let encoderLayers: [TransformerEncoderLayer]

    let regularizationValue: TangentVector {
        TangentVector(
            tokenEmbedding: tokenEmbedding.regularizationValue,
            tokenTypeEmbedding: tokenTypeEmbedding.regularizationValue,
            positionEmbedding: positionEmbedding.regularizationValue,
            embeddingLayerNorm: embeddingLayerNorm.regularizationValue,
            embeddingProjection: [Dense<Scalar>].TangentVector(
                embeddingProjection.map { $0.regularizationValue),
            encoderLayers: [TransformerEncoderLayer].TangentVector(
                encoderLayers.map { $0.regularizationValue))


    /// TODO: [DOC] Add a documentation string and fix the parameter descriptions.
    ///
    /// - Parameters:
    ///   - hiddenSize: Size of the encoder and the pooling layers.
    ///   - hiddenLayerCount: Number of hidden layers in the encoder.
    ///   - attentionHeadCount: Number of attention heads for each encoder attention layer.
    ///   - intermediateSize: Size of the encoder "intermediate" (i.e., feed-forward) layer.
    ///   - intermediateactivation= Activation function used in the encoder and the pooling layers.
    ///   - hiddenDropoutProbability: Dropout probability for all fully connected layers in the
    ///     embeddings, the encoder, and the pooling layers.
    ///   - attentionDropoutProbability: Dropout probability for the attention scores.
    ///   - maxSequenceLength: Maximum sequence length that this model might ever be used with.
    ///     Typically, this is set to something large, just in case (e.g., 512, 1024, or 2048).
    ///   - typeVocabularySize: Vocabulary size for the token type IDs passed into the BERT model.
    ///   - initializerStandardDeviation: Standard deviation of the truncated Normal initializer
    ///     used for initializing all weight matrices.
    public init(
        variant: Variant,
        vocabulary: Vocabulary,
        tokenizer: Tokenizer,
        caseSensitive: bool,
        hiddenSize: int = 768,
        hiddenLayerCount: int = 12,
        attentionHeadCount: int = 12,
        intermediateSize: int = 3072,
        intermediateactivation= @escaping Activation<Scalar> = gelu,
        hiddenDropoutProbability: Scalar = 0.1,
        attentionDropoutProbability: Scalar = 0.1,
        maxSequenceLength: int = 512,
        typeVocabularySize: int = 2,
        initializerStandardDeviation: Scalar = 0.02,
        useOneHotEmbeddings: bool = false
    ) = 
        self.variant = variant
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.caseSensitive = caseSensitive
        self.hiddenSize = hiddenSize
        self.hiddenLayerCount = hiddenLayerCount
        self.attentionHeadCount = attentionHeadCount
        self.intermediateSize = intermediateSize
        self.intermediateActivation = intermediateActivation
        self.hiddenDropoutProbability = hiddenDropoutProbability
        self.attentionDropoutProbability = attentionDropoutProbability
        self.maxSequenceLength = maxSequenceLength
        self.typeVocabularySize = typeVocabularySize
        self.initializerStandardDeviation = initializerStandardDeviation

        if case let .albert(_, hiddenGroupCount) = variant then
            precondition(
                hiddenGroupCount <= hiddenLayerCount,
                "The number of hidden groups must be smaller than the number of hidden layers.")


        let embeddingSize: int = {
            match variant with
            case .bert, .roberta, .electra: return hiddenSize
            case let .albert(embeddingSize, _): return embeddingSize

()

        self.tokenEmbedding = Embedding<Scalar>(
            vocabularySize: vocabulary.count,
            embeddingSize: embeddingSize,
            embeddingsInitializer: truncatedNormalInitializer(
                standardDeviation: Tensor<Scalar>(initializerStandardDeviation)))

        // The token type vocabulary will always be small and so we use the one-hot approach here
        // as it is always faster for small vocabularies.
        self.tokenTypeEmbedding = Embedding<Scalar>(
            vocabularySize: typeVocabularySize,
            embeddingSize: embeddingSize,
            embeddingsInitializer: truncatedNormalInitializer(
                standardDeviation: Tensor<Scalar>(initializerStandardDeviation)))

        // Since the position embeddings table is a learned variable, we create it using a (long)
        // sequence length, `maxSequenceLength`. The actual sequence length might be shorter than
        // this, for faster training of tasks that do not have long sequences. So,
        // `positionEmbedding` effectively contains an embedding table for positions
        // [0, 1, 2, ..., maxPositionEmbeddings - 1], and the current sequence may have positions
        // [0, 1, 2, ..., sequenceLength - 1], so we can just perform a slice.
        let positionPaddingIndex = { () = Int in
            match variant with
            case .bert, .albert, .electra: return 0
            | .roberta -> return 2

()
        self.positionEmbedding = Embedding(
            vocabularySize: positionPaddingIndex + maxSequenceLength,
            embeddingSize: embeddingSize,
            embeddingsInitializer: truncatedNormalInitializer(
                standardDeviation: dsharp.tensor(initializerStandardDeviation)))

        self.embeddingLayerNorm = LayerNorm<Scalar>(
            featureCount: hiddenSize,
            axis: -1)
        // TODO: Make dropout generic over the probability type.
        self.embeddingDropout = Dropout(probability: Double(hiddenDropoutProbability))

        // Add an embedding projection layer if using the ALBERT variant.
        self.embeddingProjection = {
            match variant with
            case .bert, .roberta, .electra: return []
            case let .albert(embeddingSize, _):
                // TODO: [AD] Change to optional once supported.
                return [Dense<Scalar>(
                    inputSize= embeddingSize,
                    outputSize=hiddenSize,
                    weightInitializer: truncatedNormalInitializer(
                        standardDeviation: dsharp.tensor(initializerStandardDeviation)))]

()

        match variant with
        case .bert, .roberta, .electra:
        self.encoderLayers = (0..<hiddenLayerCount).map { _ in
            TransformerEncoderLayer(
                hiddenSize: hiddenSize,
                attentionHeadCount: attentionHeadCount,
                attentionQueryactivation= { $0,
                attentionKeyactivation= { $0,
                attentionValueactivation= { $0,
                intermediateSize: intermediateSize,
                intermediateactivation= intermediateActivation,
                hiddenDropoutProbability: hiddenDropoutProbability,
                attentionDropoutProbability: attentionDropoutProbability)

        case let .albert(_, hiddenGroupCount):
        self.encoderLayers = (0..<hiddenGroupCount).map { _ in
            TransformerEncoderLayer(
                hiddenSize: hiddenSize,
                attentionHeadCount: attentionHeadCount,
                attentionQueryactivation= { $0,
                attentionKeyactivation= { $0,
                attentionValueactivation= { $0,
                intermediateSize: intermediateSize,
                intermediateactivation= intermediateActivation,
                hiddenDropoutProbability: hiddenDropoutProbability,
                attentionDropoutProbability: attentionDropoutProbability)




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
    let preprocess(sequences: [String], maxSequenceLength: int? = nil) = TextBatch {
        let maxSequenceLength = maxSequenceLength ?? self.maxSequenceLength
        let sequences = sequences.map(tokenizer.tokenize)

        // Truncate the sequences based on the maximum allowed sequence length, while accounting
        // for the '[CLS]' token and for `sequences.count` '[SEP]' tokens. The following is a
        // simple heuristic which will truncate the longer sequence one token at a time. This makes 
        // more sense than truncating an equal percent of tokens from each sequence, since if one
        // sequence is very short then each token that is truncated likely contains more
        // information than respective tokens in longer sequences.
        let totalLength = sequences.map { $0.count.reduce(0, +)
        let totalLengthLimit = { () = Int in
            match variant with
            case .bert, .albert, .electra: return maxSequenceLength - 1 - sequences.count
            | .roberta -> return maxSequenceLength - 1 - 2 * sequences.count

()
        while totalLength >= totalLengthLimit {
            let maxIndex = sequences.enumerated().max(by: { $0.1.count < $1.1.count)!.0
            sequences[maxIndex] = [String](sequences[maxIndex].dropLast())
            totalLength = sequences.map { $0.count.reduce(0, +)


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
        let tokens = ["[CLS]"]
        let tokenTypeIds = [int32(0)]
        for (sequenceId, sequence) in sequences.enumerated() = 
            for token in sequence do
                tokens.append(token)
                tokenTypeIds.append(int32(sequenceId))

            tokens.append("[SEP]")
            tokenTypeIds.append(int32(sequenceId))
            if case .roberta = variant, sequenceId < sequences.count - 1 then
                tokens.append("[SEP]")
                tokenTypeIds.append(int32(sequenceId))


        let tokenIds = tokens.map { int32(vocabulary.id(forToken: $0)!)

        // The mask is set to `true` for real tokens and `false` for padding tokens. This is so
        // that only real tokens are attended to.
        let mask = [int32](repeating: 1, count: tokenIds.count)

        return TextBatch(
            tokenIds: dsharp.tensor(tokenIds).unsqueeze(0),
            tokenTypeIds: dsharp.tensor(tokenTypeIds).unsqueeze(0),
            mask: dsharp.tensor(mask).unsqueeze(0))


    (wrt: self)
    override _.forward(input: TextBatch) : Tensor =
        let sequenceLength = input.tokenIds.shape.[1]
        let variant = withoutDerivative(at: self.variant)

        // Compute the input embeddings and apply layer normalization and dropout on them.
        let tokenEmbeddings = tokenEmbedding(input.tokenIds)
        let tokenTypeEmbeddings = tokenTypeEmbedding(input.tokenTypeIds)
        let positionPaddingIndex: int
        match variant with
        case .bert, .albert, .electra: positionPaddingIndex = 0
        | .roberta -> positionPaddingIndex = 2

        let positionEmbeddings = positionEmbedding.embeddings.slice(
            lowerBounds: [positionPaddingIndex, 0],
            upperBounds: [positionPaddingIndex + sequenceLength, -1]
        ).unsqueeze(0)
        let embeddings = tokenEmbeddings + positionEmbeddings

        // Add token type embeddings if needed, based on which BERT variant is being used.
        match variant with
        case .bert, .albert, .electra: embeddings = embeddings + tokenTypeEmbeddings
        | .roberta -> break


        embeddings = embeddingLayerNorm(embeddings)
        embeddings = embeddingDropout(embeddings)

        if case .albert = variant then
            embeddings = embeddingProjection[0](embeddings)


        // Create an attention mask for the inputs with shape
        // `[batchSize, sequenceLength, sequenceLength]`.
        let attentionMask = createAttentionMask(forTextBatch: input)

        // We keep the representation as a 2-D tensor to avoid reshaping it back and forth from a
        // 3-D tensor to a 2-D tensor. Reshapes are normally free on GPUs/CPUs but may not be free
        // on TPUs, and so we want to minimize them to help the optimizer.
        let transformerInput = embeddings.reshapedToMatrix()
        let batchSize = embeddings.shape.[0]

        // Run the stacked transformer.
        match variant with
        case .bert, .roberta, .electra:
            for layerIndex in 0..<(withoutDerivative(at: encoderLayers) =  $0.count) = 
                transformerInput = encoderLayers[layerIndex](TransformerInput(
                sequence: transformerInput,
                attentionMask: attentionMask,
                batchSize= batchSize))

        case let .albert(_, hiddenGroupCount):
            let groupsPerLayer = double(hiddenGroupCount) / double(hiddenLayerCount)
            for layerIndex in 0..<hiddenLayerCount {
                let groupIndex = int(double(layerIndex) * groupsPerLayer)
                transformerInput = encoderLayers[groupIndex](TransformerInput(
                    sequence: transformerInput,
                    attentionMask: attentionMask,
                    batchSize= batchSize))



        // Reshape back to the original tensor shape.
        return transformerInput.reshapedFromMatrix(originalShape: embeddings.shape)



extension BERT {
    type Variant: CustomStringConvertible {
        /// - Source: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](
        ///             https://arxiv.org/pdf/1810.04805.pdf).
        case bert

        /// - Source: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](
        ///             https://arxiv.org/pdf/1907.11692.pdf).
        case roberta

        /// - Source: [ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations](
        ///             https://arxiv.org/pdf/1909.11942.pdf).
        case albert(embeddingSize: int, hiddenGroupCount: int)

        /// - Source: [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators]
        ///              https://arxiv.org/abs/2003.10555
        case electra

        let description: string {
            match self with
            | .bert ->
                return "bert"
            | .roberta ->
                return "roberta"
            case let .albert(embeddingSize, hiddenGroupCount):
                return "albert-E-\(embeddingSize)-G-\(hiddenGroupCount)"
            | .electra ->
                return "electra"





//===-----------------------------------------------------------------------------------------===//
// Tokenization
//===-----------------------------------------------------------------------------------------===//

/// BERT tokenizer that is simply defined as the composition of the basic text tokenizer and the
/// greedy subword tokenizer.
type BERTTokenizer: Tokenizer {
    let caseSensitive: bool
    let vocabulary: Vocabulary
    let unknownToken: string
    let maxTokenLength: int?

    let basicTextTokenizer: BasicTokenizer
    let greedySubwordTokenizer: GreedySubwordTokenizer

    /// Creates a BERT tokenizer.
    ///
    /// - Parameters:
    ///   - vocabulary: Vocabulary containing all supported tokens.
    ///   - caseSensitive: Specifies whether or not to ignore case.
    ///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
    ///     provided vocabulary or whose length is longer than `maxTokenLength`).
    ///   - maxTokenLength: Maximum allowed token length.
    public init(
        vocabulary: Vocabulary,
        caseSensitive: bool = false,
        unknownToken: string = "[UNK]",
        maxTokenLength: int? = nil
    ) = 
        self.caseSensitive = caseSensitive
        self.vocabulary = vocabulary
        self.unknownToken = unknownToken
        self.maxTokenLength = maxTokenLength
        self.basicTextTokenizer = BasicTokenizer(caseSensitive: caseSensitive)
        self.greedySubwordTokenizer = GreedySubwordTokenizer(
            vocabulary: vocabulary,
            unknownToken: unknownToken,
            maxTokenLength: maxTokenLength)


    let tokenize(_ text: string) = [String] {
        basicTextTokenizer.tokenize(text).flatMap(greedySubwordTokenizer.tokenize)



/// RoBERTa tokenizer that is simply defined as the composition of the basic text tokenizer and the
/// byte pair encoder.
type RoBERTaTokenizer: Tokenizer {
    let caseSensitive: bool
    let unknownToken: string

    let bytePairEncoder: BytePairEncoder

    let tokenizationRegex: NSRegularExpression = try! NSRegularExpression(
        pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L+| ?\\p{N+| ?[^\\s\\p{L\\p{N]+|\\s+(?!\\S)|\\s+")

    /// Creates a full text tokenizer.
    ///
    /// - Parameters:
    ///   - bytePairEncoder: Byte pair encoder to use.
    ///   - caseSensitive: Specifies whether or not to ignore case.
    ///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
    ///     provided vocabulary or whose length is longer than `maxTokenLength`).
    public init(
        bytePairEncoder: BytePairEncoder,
        caseSensitive: bool = false,
        unknownToken: string = "[UNK]"
    ) = 
        self.caseSensitive = caseSensitive
        self.unknownToken = unknownToken
        self.bytePairEncoder = bytePairEncoder


    let tokenize(_ text: string) = [String] {
        let matches = tokenizationRegex.matches(
            in: text,
            range: NSRange(text.startIndex..., in: text))
        return matches.flatMap { match -> [String] in
            if let range = Range(match.range, in: text) = 
                return bytePairEncoder.encode(token: string(text[range]))
            else
                return []





//===-----------------------------------------------------------------------------------------===//
// Pre-Trained Models
//===-----------------------------------------------------------------------------------------===//

extension BERT {
    type PreTrainedModel {
        case bertBase(cased: bool, multilingual: bool)
        case bertLarge(cased: bool, wholeWordMasking: bool)
        case robertaBase
        case robertaLarge
        case albertBase
        case albertLarge
        case albertXLarge
        case albertXXLarge
        case electraBase
        case electraLarge

        /// The name of this pre-trained model.
        let name: string {
            match self with
            case .bertBase(false, false): return "BERT Base Uncased"
            case .bertBase(true, false): return "BERT Base Cased"
            case .bertBase(false, true): return "BERT Base Multilingual Uncased"
            case .bertBase(true, true): return "BERT Base Multilingual Cased"
            case .bertLarge(false, false): return "BERT Large Uncased"
            case .bertLarge(true, false): return "BERT Large Cased"
            case .bertLarge(false, true): return "BERT Large Whole-Word-Masking Uncased"
            case .bertLarge(true, true): return "BERT Large Whole-Word-Masking Cased"
            | .robertaBase -> return "RoBERTa Base"
            | .robertaLarge -> return "RoBERTa Large"
            | .albertBase -> return "ALBERT Base"
            | .albertLarge -> return "ALBERT Large"
            | .albertXLarge -> return "ALBERT xLarge"
            | .albertXXLarge -> return "ALBERT xxLarge"
            | .electraBase -> return "ELECTRA Base"
            | .electraLarge -> return "ELECTRA Large"



        /// The URL where this pre-trained model can be downloaded from.
        let url: Uri {
            let bertPrefix = "https://storage.googleapis.com/bert_models/2018_"
            let robertaPrefix = "https://storage.googleapis.com/s4tf-hosted-binaries/checkpoints/Text/RoBERTa"
            let albertPrefix = "https://storage.googleapis.com/tfhub-modules/google/albert"
            let electraPrefix = "https://storage.googleapis.com/electra-data/electra_"
            match self with
            case .bertBase(false, false):
                return Uri("\(bertPrefix)10_18/\(subDirectory).zip")!
            case .bertBase(true, false):
                return Uri("\(bertPrefix)10_18/\(subDirectory).zip")!
            case .bertBase(false, true):
                return Uri("\(bertPrefix)11_03/\(subDirectory).zip")!
            case .bertBase(true, true):
                return Uri("\(bertPrefix)11_23/\(subDirectory).zip")!
            case .bertLarge(false, false):
                return Uri("\(bertPrefix)10_18/\(subDirectory).zip")!
            case .bertLarge(true, false):
                return Uri("\(bertPrefix)10_18/\(subDirectory).zip")!
            case .bertLarge(false, true):
                return Uri("\(bertPrefix)05_30/\(subDirectory).zip")!
            case .bertLarge(true, true):
                return Uri("\(bertPrefix)05_30/\(subDirectory).zip")!
            | .robertaBase ->
                return Uri("\(robertaPrefix)/base.zip")!
            | .robertaLarge ->
                return Uri("\(robertaPrefix)/large.zip")!
            case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                return Uri("\(albertPrefix)_\(subDirectory)/1.tar.gz")!
            | .electraBase ->
                return Uri("\(electraPrefix)base.zip")!
            | .electraLarge ->
                return Uri("\(electraPrefix)large.zip")!



        let variant: Variant {
            match self with
            case .bertBase, .bertLarge:
                return .bert
            case .robertaBase, .robertaLarge:
                return .roberta
            case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                return .albert(embeddingSize: 128, hiddenGroupCount: 1)
            case .electraBase, .electraLarge:
                return .electra



        let caseSensitive: bool {
            match self with
            case let .bertBase(cased, _): return cased
            case let .bertLarge(cased, _): return cased
            case .robertaBase, .robertaLarge: return true
            case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge: return false
            case .electraBase, .electraLarge: return false



        let hiddenSize: int {
            match self with
            | .bertBase -> return 768
            | .bertLarge -> return 1024
            | .robertaBase -> return 768
            | .robertaLarge -> return 1024
            | .albertBase -> return 768
            | .albertLarge -> return 1024
            | .albertXLarge -> return 2048
            | .albertXXLarge -> return 4096
            | .electraBase -> return 768
            | .electraLarge -> return 1024



        let hiddenLayerCount: int {
            match self with
            | .bertBase -> return 12
            | .bertLarge -> return 24
            | .robertaBase -> return 12
            | .robertaLarge -> return 24
            | .albertBase -> return 12
            | .albertLarge -> return 24
            | .albertXLarge -> return 24
            | .albertXXLarge -> return 12
            | .electraBase -> return 12
            | .electraLarge -> return 24



        let attentionHeadCount: int {
            match self with
            | .bertBase -> return 12
            | .bertLarge -> return 16
            | .robertaBase -> return 12
            | .robertaLarge -> return 16
            | .albertBase -> return 12
            | .albertLarge -> return 16
            | .albertXLarge -> return 16
            | .albertXXLarge -> return 64
            | .electraBase -> return 12
            | .electraLarge -> return 16



        let intermediateSize: int {
            match self with
            | .bertBase -> return 3072
            | .bertLarge -> return 4096
            | .robertaBase -> return 3072
            | .robertaLarge -> return 4096
            | .albertBase -> return 3072
            | .albertLarge -> return 4096
            | .albertXLarge -> return 8192
            | .albertXXLarge -> return 16384
            | .electraBase -> return 3072
            | .electraLarge -> return 4096



        /// The sub-directory of this pre-trained model.
        internal let subDirectory: string {
            match self with
            case .bertBase(false, false): return "uncased_L-12_H-768_A-12"
            case .bertBase(true, false): return "cased_L-12_H-768_A-12"
            case .bertBase(false, true): return "multilingual_L-12_H-768_A-12"
            case .bertBase(true, true): return "multi_cased_L-12_H-768_A-12"
            case .bertLarge(false, false): return "uncased_L-24_H-1024_A-16"
            case .bertLarge(true, false): return "cased_L-24_H-1024_A-16"
            case .bertLarge(false, true): return "wwm_uncased_L-24_H-1024_A-16"
            case .bertLarge(true, true): return "wwm_cased_L-24_H-1024_A-16"
            | .robertaBase -> return "base"
            | .robertaLarge -> return "large"
            | .albertBase -> return "base"
            | .albertLarge -> return "large"
            | .albertXLarge -> return "xLarge"
            | .albertXXLarge -> return "xxLarge"
            | .electraBase -> return "electra_base"
            | .electraLarge -> return "electra_large"



        /// Loads this pre-trained BERT model from the specified URL.
        ///
        /// - Note: This function will download the pre-trained model files to the specified
        //    directory, if they are not already there.
        ///
        /// - Parameters:
        ///   - url: Uri to load the pretrained model from.
        let load(from url: Uri? = nil) -> BERT {
            print("Loading BERT pre-trained model '\(name)'.")
            
            let reader = try CheckpointReader(checkpointLocation: url ?? self.url, modelName: name)
            // TODO(michellecasbon): expose this.
            reader.isCRCVerificationEnabled = false

            let storage = reader.localCheckpointLocation.deletingLastPathComponent()

            // Load the appropriate vocabulary file.
            let vocabulary: Vocabulary = {
                match self with
                case .bertBase, .bertLarge, .electraBase, .electraLarge:
                    let vocabularyURL = storage </> ("vocab.txt")
                    return try! Vocabulary(fromFile: vocabularyURL)
                case .robertaBase, .robertaLarge:
                    let vocabularyURL = storage </> ("vocab.json")
                    let dictionaryURL = storage </> ("dict.txt")
                    return try! Vocabulary(
                        fromRoBERTaJSONFile: vocabularyURL,
                        dictionaryFile: dictionaryURL)
                case .albertBase, .albertLarge, .albertXLarge, .albertXXLarge:
                    let vocabularyURL = storage
                        .deletingLastPathComponent()
                         </> ("assets")
                         </> ("30k-clean.model")
                    return try! Vocabulary(fromSentencePieceModel: vocabularyURL)

()

            // Create the tokenizer and load any necessary files.
            let tokenizer: Tokenizer = try {
                match self with
                case .bertBase, .bertLarge, .albertBase, .albertLarge, .albertXLarge, .albertXXLarge,
                .electraBase, .electraLarge:
                    return BERTTokenizer(
                        vocabulary: vocabulary,
                        caseSensitive: caseSensitive,
                        unknownToken: "[UNK]",
                        maxTokenLength: nil)
                case .robertaBase, .robertaLarge:
                    let mergePairsFileURL = storage
                         </> ("merges.txt")
                    let mergePairs = [BytePairEncoder.Pair: int](
                        uniqueKeysWithValues:
                            (try String(contentsOfFile: mergePairsFileURL.path, encoding: .utf8))
                                .components(separatedBy: .newlines)
                                .dropFirst()
                                .enumerated()
                                .compactMap { (index, line) = (BytePairEncoder.Pair, Int)? in
                                    let lineParts = line.split(separator: " ")
                                    if lineParts.count < 2 then return nil
                                    return (
                                        BytePairEncoder.Pair(
                                            String(lineParts[0]),
                                            String(lineParts[1])),
                                        index)
)
                    return RoBERTaTokenizer(
                        bytePairEncoder: BytePairEncoder(
                            vocabulary: vocabulary,
                            mergePairs: mergePairs),
                        caseSensitive: caseSensitive,
                        unknownToken: "[UNK]")

()

            // Create a BERT model.
            let model = BERT(
                variant: variant,
                vocabulary: vocabulary,
                tokenizer: tokenizer,
                caseSensitive: caseSensitive,
                hiddenSize: hiddenSize,
                hiddenLayerCount: hiddenLayerCount,
                attentionHeadCount: attentionHeadCount,
                intermediateSize: intermediateSize,
                intermediateactivation= gelu,
                hiddenDropoutProbability: 0.1,
                attentionDropoutProbability: 0.1,
                maxSequenceLength: 512,
                typeVocabularySize: 2,
                initializerStandardDeviation: 0.02,
                useOneHotEmbeddings: false)

            model.loadTensors(reader)
            return model



    /// Loads a BERT model from the provided CheckpointReader into this BERT model.
    ///
    /// - Parameters:
    ///   - reader: CheckpointReader object to load tensors from.
    public mutating let loadTensors(_ reader: CheckpointReader) = 
        match variant with
        case .bert, .albert, .roberta:    
            tokenEmbedding.embeddings =
                reader.readTensor(name= "bert/embeddings/word_embeddings")
            positionEmbedding.embeddings =
                reader.readTensor(name= "bert/embeddings/position_embeddings")
            embeddingLayerNorm.offset =
                reader.readTensor(name= "bert/embeddings/LayerNorm/beta")
            embeddingLayerNorm.scale =
                reader.readTensor(name= "bert/embeddings/LayerNorm/gamma")
        | .electra ->
            tokenEmbedding.embeddings =
                reader.readTensor(name= "electra/embeddings/word_embeddings")
            positionEmbedding.embeddings =
                reader.readTensor(name= "electra/embeddings/position_embeddings")
            embeddingLayerNorm.offset =
                reader.readTensor(name= "electra/embeddings/LayerNorm/beta")
            embeddingLayerNorm.scale =
                reader.readTensor(name= "electra/embeddings/LayerNorm/gamma")

        match variant with
        case .bert, .albert:
            tokenTypeEmbedding.embeddings =
                reader.readTensor(name= "bert/embeddings/token_type_embeddings")
        | .roberta -> ()
        | .electra ->
            tokenTypeEmbedding.embeddings =
                reader.readTensor(name= "electra/embeddings/token_type_embeddings")    

        match variant with
        case .bert, .roberta:
            for layerIndex in encoderLayers.indices do
                encoderLayers[layerIndex].load(bert: reader,
                    prefix: "bert/encoder/layer_\(layerIndex)")

        | .albert ->
            embeddingProjection[0].weight =
                reader.readTensor(name= "bert/encoder/embedding_hidden_mapping_in/kernel")
            embeddingProjection[0].bias =
                reader.readTensor(name= "bert/encoder/embedding_hidden_mapping_in/bias")
            for layerIndex in encoderLayers.indices do
                let prefix = "bert/encoder/transformer/group_\(layerIndex)/inner_group_0"
                encoderLayers[layerIndex].load(albert: reader, prefix: prefix)

        | .electra ->
            for layerIndex in encoderLayers.indices do
                let prefix = "electra/encoder/layer_\(layerIndex)"
                encoderLayers[layerIndex].multiHeadAttention.queryWeight =
                    reader.readTensor(name=  "\(prefix)/attention/self/query/kernel")
                encoderLayers[layerIndex].multiHeadAttention.queryBias =
                    reader.readTensor(name=  "\(prefix)/attention/self/query/bias")
                encoderLayers[layerIndex].multiHeadAttention.keyWeight =
                    reader.readTensor(name=  "\(prefix)/attention/self/key/kernel")
                encoderLayers[layerIndex].multiHeadAttention.keyBias =
                    reader.readTensor(name=  "\(prefix)/attention/self/key/bias")
                encoderLayers[layerIndex].multiHeadAttention.valueWeight =
                    reader.readTensor(name=  "\(prefix)/attention/self/value/kernel")
                encoderLayers[layerIndex].multiHeadAttention.valueBias =
                    reader.readTensor(name=  "\(prefix)/attention/self/value/bias")
                encoderLayers[layerIndex].attentionWeight =
                    reader.readTensor(name=  "\(prefix)/attention/output/dense/kernel")
                encoderLayers[layerIndex].attentionBias =
                    reader.readTensor(name=  "\(prefix)/attention/output/dense/bias")
                encoderLayers[layerIndex].attentionLayerNorm.offset =
                    reader.readTensor(name=  "\(prefix)/attention/output/LayerNorm/beta")
                encoderLayers[layerIndex].attentionLayerNorm.scale =
                    reader.readTensor(name=  "\(prefix)/attention/output/LayerNorm/gamma")
                encoderLayers[layerIndex].intermediateWeight =
                    reader.readTensor(name=  "\(prefix)/intermediate/dense/kernel")
                encoderLayers[layerIndex].intermediateBias =
                    reader.readTensor(name=  "\(prefix)/intermediate/dense/bias")
                encoderLayers[layerIndex].outputWeight =
                    reader.readTensor(name=  "\(prefix)/output/dense/kernel")
                encoderLayers[layerIndex].outputBias =
                    reader.readTensor(name=  "\(prefix)/output/dense/bias")
                encoderLayers[layerIndex].outputLayerNorm.offset =
                    reader.readTensor(name=  "\(prefix)/output/LayerNorm/beta")
                encoderLayers[layerIndex].outputLayerNorm.scale =
                    reader.readTensor(name=  "\(prefix)/output/LayerNorm/gamma")





extension Vocabulary {
    internal init(fromRoBERTaJSONFile fileURL: Uri, dictionaryFile dictionaryURL: Uri) =
        let dictionary = [Int: int](
            uniqueKeysWithValues:
                (try String(contentsOfFile: dictionaryURL.path, encoding: .utf8))
                    .components(separatedBy: .newlines)
                    .compactMap { line in
                        let lineParts = line.split(separator: " ")
                        if lineParts.count < 1 then return nil
                        return int(lineParts[0])

                    .enumerated()
                    .map { ($1, $0 + 4))
        let json = try String(contentsOfFile: fileURL.path)
        let tokensToIds = try JSONDecoder().decode(
            Map<string, int>.self,
            from: json.data(using: .utf8)!)
        tokensToIds = tokensToIds.mapValues { dictionary[$0]!
        tokensToIds.merge(["[CLS]": 0, "[PAD]": 1, "[SEP]": 2, "[UNK]": 3]) =  (_, new) in new
        self.init(tokensToIds: tokensToIds)


