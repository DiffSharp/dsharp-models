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

namespace Support

open DiffSharp

[<AutoOpen>]
module Tokenization =
    /// Returns a 3-D attention mask that correspond to the 2-D mask of the provided text batch.
    ///
    /// - Parameters:
    ///   - text: Text batch for which to create an attention mask. `input.mask` has shape
    ///     `[batchSize, sequenceLength]`.
    ///
    /// - Returns: Attention mask with shape `[batchSize, sequenceLength, sequenceLength]`.
    let createAttentionMask(text: TextBatch) : Tensor =
        let batchSize = text.tokenIds.shape.[0]
        let fromSequenceLength = text.tokenIds.shape.[1]
        let toSequenceLength = text.mask.shape.[1]
        let reshapedMask = text.mask.view([batchSize; 1; toSequenceLength])

        // We do not assume that `input.tokenIds` is a mask. We do not actually care if we attend
        // *from* padding tokens (only *to* padding tokens) so we create a tensor of all ones.
        let broadcastOnes = dsharp.ones([batchSize; fromSequenceLength; 1], device=text.mask.device)

        // We broadcast along two dimensions to create the mask.
        broadcastOnes * reshapedMask

/// Vocabulary that can be used for tokenizing strings.
type Vocabulary(tokensToIds: Map<string, int>) =
    let idsToTokens = Map<int, string>(tokensToIds |> Seq.map (fun (KeyValue(a,b)) -> (b,a)))

    member _.count = tokensToIds.Count

    new (idsToTokens: Map<int, string>) = 
        Vocabulary(Map<string, int>(idsToTokens |> Seq.map (fun (KeyValue(a,b)) -> (b,a))))

    member _.contains(token: string) =
        tokensToIds.ContainsKey(token)

    member _.id(token: string) = 
        tokensToIds.[token]

    member _.token(id: int) = 
        idsToTokens.[id]
(*
    static member FromFile (fileURL: Uri) =
        Vocabulary (
            (try String(contentsOfFile: fileURL.path, encoding: .utf8))
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines)
                .filter (fun x -> x..count > 0
                .enumerated().map { ($0.element, $0.offset),
            uniquingKeysWith: { (v1, v2) in max(v1, v2)))

    member _.save(toFile fileURL: Uri) =
        try idsToTokens
            .sorted { $0.key < $1.key
            .map { $0.1
            .joined(separator: "\n")
            .write(fileURL, atomically: true, encoding: .utf8)

    static member FromSentencePieceModel(fileURL: Uri) =
            tokensToIds: Map<string, int>(
                (try Sentencepiece_ModelProto(serializedData: Data(contentsOf: fileURL)))
                    .pieces
                    .map { $0.piece.replacingOccurrences(of: "▁", with: "##")
                    .map { $0 = "<unk>" ? "[UNK]" : $0
                    .enumerated().map { ($0.element, $0.offset),
                uniquingKeysWith: { (v1, v2) in max(v1, v2)))

    public init(fromJSONFile fileURL: Uri) =
        let json = try String(contentsOfFile: fileURL.path)
        let tokensToIds = try JSONDecoder().decode(
            Map<string, int>.self,
            from: json.data(using: .utf8)!)
        self.init(tokensToIds: tokensToIds)

/// Text tokenizer which is used to split strings into arrays of tokens.
type Tokenizer =
    abstract tokenize: _ text: string -> string[]

/// Basic text tokenizer that performs some simple preprocessing to clean the provided text and
/// then performs tokenization based on whitespaces.
type BasicTokenizer() = 
    let caseSensitive: bool

    /// Creates a basic text tokenizer.
    ///
    /// Arguments:
    ///   - caseSensitive: Specifies whether or not to ignore case.
    public init(caseSensitive: bool = false) = 
        self.caseSensitive = caseSensitive


    let tokenize(_ text: string) = [String] {
        clean(text).split(separator: " ").flatMap { token -> [String] in
            let processed = String(token)
            if !caseSensitive then
                processed = processed.lowercased()

                // Normalize unicode characters.
                processed = processed.decomposedStringWithCanonicalMapping

                // Strip accents.
                processed = processed.replacingOccurrences(
                    of: #"\p{Mn"#,
                    with e ->: "",
                    options: .regularExpression)


            // Split punctuation. We treat all non-letter/number ASCII as punctuation. Characters
            // such as "$" are not in the Unicode Punctuation class but we treat them as
            // punctuation anyways for consistency.
            processed = processed.replacingOccurrences(
                of: #"([\p{P!-/:-@\[-`{-~])"#,
                with e ->: " $1 ",
                options: .regularExpression)

            return processed.split(separator: " ").map(String.init)




/// Greedy subword tokenizer.
///
/// This tokenizer uses a greedy longest-match-first algorithm to perform tokenization using the
/// provided vocabulary. For example, `"unaffable"` could be tokenized as
/// `["un", "##aff", "##able"]`.
type GreedySubwordTokenizer: Tokenizer {
    let vocabulary: Vocabulary
    let unknownToken: string
    let maxTokenLength: int?

    /// Creates a subword tokenizer.
    ///
    /// - Parameters:
    ///   - vocabulary: Vocabulary containing all supported tokens.
    ///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
    ///     provided vocabulary or whose length is longer than `maxTokenLength`).
    ///   - maxTokenLength: Maximum allowed token length.
    public init(vocabulary: Vocabulary, unknownToken: string = "[UNK]", maxTokenLength: int?) = 
        self.vocabulary = vocabulary
        self.unknownToken = unknownToken
        self.maxTokenLength = maxTokenLength


    let tokenize(_ text: string) = [String] {
        clean(text).split(separator: " ").flatMap { token -> [String] in
            if let maxLength = maxTokenLength, token.count > maxLength then return [unknownToken]
            let isBad = false
            let start = token.startIndex
            let subTokens = [String]()
            while start < token.endIndex {
                // Find the longest matching substring.
                let end = token.endIndex
                let currentSubstring = ""
                while start < end {
                    let substring = String(token[start..<end])
                    if start > token.startIndex then
                        substring = "##" + substring

                    if vocabulary.contains(substring) = 
                        currentSubstring = substring
                        start = end
                    else
                        end = token.index(end, offsetBy: -1)



                // Check if the substring is good.
                if currentSubstring.isEmpty then
                    isBad = true
                    start = token.endIndex
                else
                    subTokens.append(currentSubstring)
                    start = end


            return isBad ? [unknownToken] : subTokens




/// Returns a cleaned version of the provided string. Cleaning in this case consists of normalizing
/// whitespaces, removing control characters and adding whitespaces around CJK characters.
///
/// - Parameters:
///   - text: string to clean.
///
/// - Returns: Cleaned version of `text`.
internal let clean(_ text: string) =
    // Normalize whitespaces.
    let afterWhitespace = text.replacingOccurrences(
        of: #"\s+"#,
        with e ->: " ",
        options: .regularExpression)

    // Remove control characters.
    let afterControl = afterWhitespace.replacingOccurrences(
        of: #"[\x{0000\x{fffd\p{C]"#,
        with e ->: "",
        options: .regularExpression)

    // Add whitespace around CJK characters.
    //
    // The regular expression that we use defines a "chinese character" as anything in the
    // [CJK Unicode block](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)).
    //
    // Note that the CJK Unicode block is not all Japanese and Korean characters, despite its name.
    // The modern Korean Hangul alphabet is a different block, as is Japanese Hiragana and
    // Katakana. Those alphabets are used to write space-separated words, and so they are not
    // treated specially and are instead handled like all of the other languages.
    let afterCJK = afterControl.replacingOccurrences(
        of: #"([\p{InCJK_Unified_Ideographs"# +
            #"\p{InCJK_Unified_Ideographs_Extension_A"# +
            #"\p{InCJK_Compatibility_Ideographs"# +
            #"\x{20000-\x{2a6df"# +
            #"\x{2a700-\x{2b73f"# +
            #"\x{2b740-\x{2b81f"# +
            #"\x{2b820-\x{2ceaf"# +
            #"\x{2f800-\x{2fa1f])"#,
        with e ->: " $1 ",
        options: .regularExpression)

    return afterCJK


*)
