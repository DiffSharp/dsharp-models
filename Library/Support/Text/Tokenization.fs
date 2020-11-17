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

open System
open System.IO
open System.Text
open System.Text.Json
open System.Text.RegularExpressions
open DiffSharp

type Data(bytes: byte[]) =

    static member ReadAllBytes(file) = Data(File.ReadAllBytes(file))


[<AutoOpen>]
module Tokenization =

    type String with

        member t.ReplaceRegexp(regexp: string, replacement: string) : string =
            failwith "tbd - check syntax of all regexps"
            Regex.Replace(t, regexp, replacement)

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

    /// Returns a cleaned version of the provided string. Cleaning in this case consists of normalizing
    /// whitespaces, removing control characters and adding whitespaces around CJK characters.
    ///
    /// - Parameters:
    ///   - text: string to clean.
    ///
    /// - Returns: Cleaned version of `text`.
    let clean(text: string) : string =
        // Normalize whitespaces.
        let afterWhitespace : string = text.ReplaceRegexp(@"\s+"," ")

        // Remove control characters.
        let afterControl : string = afterWhitespace.ReplaceRegexp(@"[\x{0000}\x{fffd}\p{C]}]", "")

        // Add whitespace around CJK characters.
        //
        // The regular expression that we use defines a "chinese character" as anything in the
        // [CJK Unicode block](https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)).
        //
        // Note that the CJK Unicode block is not all Japanese and Korean characters, despite its name.
        // The modern Korean Hangul alphabet is a different block, as is Japanese Hiragana and
        // Katakana. Those alphabets are used to write space-separated words, and so they are not
        // treated specially and are instead handled like all of the other languages.
        let afterCJK : string = 
            afterControl.ReplaceRegexp(
                @"([\p{InCJK_Unified_Ideographs}" +
                @"\p{InCJK_Unified_Ideographs_Extension_A}" +
                @"\p{InCJK_Compatibility_Ideographs}" +
                @"\x{20000}-\x{2a6df}" +
                @"\x{2a700-\x{2b73f}" +
                @"\x{2b740-\x{2b81f}" +
                @"\x{2b820-\x{2ceaf}" +
                @"\x{2f800-\x{2fa1f])",
                " $1 ")

        afterCJK


/// Vocabulary that can be used for tokenizing strings.
type Vocabulary(tokensToIds: Map<string, int>) =
    let idsToTokens = Map<int, string>(tokensToIds |> Seq.map (fun (KeyValue(a,b)) -> (b,a)))

    member _.count = tokensToIds.Count

    new (idsToTokens: Map<int, string>) = 
        Vocabulary(idsToTokens |> Seq.map (fun (KeyValue(a,b)) -> (b,a)) |> Map.ofSeq)

    member _.contains(token: string) =
        tokensToIds.ContainsKey(token)

    member _.id(token: string) = 
        tokensToIds.[token]

    member _.token(id: int) = 
        idsToTokens.[id]

    static member FromFile (fileURL: FilePath) =
        File.ReadAllLines(fileURL)
        |> Array.map (fun x -> x.Trim())
        |> Array.filter (fun x -> x.Length > 0)
        |> Array.mapi (fun x i -> x, i)
        |> Map.ofSeq
        |> Vocabulary

    member _.save(fileURL: FilePath) =
        idsToTokens
        |> Seq.sortBy (fun x -> x.Key)
        |> Seq.map (fun x -> x.Value)
        |> String.concat "\n"
        |> fun s -> File.WriteAllText(fileURL, s)

    static member FromSentencePieceModel(fileURL: FilePath) : Vocabulary = failwith "tbd"
    //    let data = Data.ReadAllBytes(fileURL)
    //    Vocabulary(
    //        (Sentencepiece_ModelProto(data))
    //            .pieces
    //            .map(fun x -> x.piece.Replace("▁", "@"))
    //            .map(fun x -> if x = "<unk>" then "[UNK]" else x)
    //            |> Seq.map(fun x -> (x.element, x.offset)),
    //        uniquingKeysWith=(fun (v1, v2) -> max(v1, v2)))

    static member FromJsonFile(fileURL: FilePath) =
        let json = File.ReadAllText(fileURL)
        let j = Json.JsonDocument.Parse(json)
        //let tokensToIds =  JSONDecoder().decode(Map<string, int>.self, from: json.data(using: .utf8)!)
        //Vocabulary(tokensToIds)
        failwith "tbd"

/// Text tokenizer which is used to split strings into arrays of tokens.
[<AbstractClass>]
type Tokenizer() =
    abstract tokenize: text: string -> string[]

/// Basic text tokenizer that performs some simple preprocessing to clean the provided text and
/// then performs tokenization based on whitespaces.
///
/// Creates a basic text tokenizer.
///
/// Arguments:
///   - caseSensitive: Specifies whether or not to ignore case.
type BasicTokenizer(?caseSensitive: bool) = 
    inherit Tokenizer()
    
    let caseSensitive = defaultArg caseSensitive false

    override _.tokenize(text: string) : string[] =
        clean(text).Split(" ") |> Array.collect (fun token ->
            let mutable processed = token
            if not caseSensitive then
                processed <- processed.ToLower()

                // Normalize unicode characters.
                processed <- processed.Normalize(NormalizationForm.FormD)

                // Strip accents.
                processed <- processed.ReplaceRegexp(@"\p{Mn}", "")

            // Split punctuation. We treat all non-letter/number ASCII as punctuation. Characters
            // such as "$" are not in the Unicode Punctuation class but we treat them as
            // punctuation anyways for consistency.
            processed <- processed.ReplaceRegexp(@"([\p{P}!-/:-@\[-`{-~])"," $1 ")

            processed.Split(" "))

/// Greedy subword tokenizer.
///
/// This tokenizer uses a greedy longest-match-first algorithm to perform tokenization using the
/// provided vocabulary. For example, `"unaffable"` could be tokenized as
/// `["un", "#aff", "#able"]`.
///
/// Creates a subword tokenizer.
///
/// - Parameters:
///   - vocabulary: Vocabulary containing all supported tokens.
///   - unknownToken: Token used to represent unknown tokens (i.e., tokens that are not in the
///     provided vocabulary or whose length is longer than `maxTokenLength`).
///   - maxTokenLength: Maximum allowed token length.
type GreedySubwordTokenizer(vocabulary: Vocabulary, ?unknownToken: string , ?maxTokenLength: int) =
    inherit Tokenizer() 
    let unknownToken = defaultArg unknownToken "[UNK]"
    override _.tokenize(text) =
        [| for token in clean(text).Split(" ") do 
            let maxLength = defaultArg maxTokenLength token.Length
            if token.Length > maxLength then 
                yield unknownToken
            else
                let mutable isBad = false
                let mutable start = 0
                let subTokens = ResizeArray()
                while start < token.Length do
                    // Find the longest matching substring.
                    let mutable fin = token.Length
                    let mutable currentSubstring = ""
                    while start < fin do
                        let mutable substring = token.[start..fin-1]
                        if start > 0 then
                            substring <- "@" + substring

                        if vocabulary.contains(substring) then
                            currentSubstring <- substring
                            start <- fin
                        else
                            failwith "tbd"
                            //fin <- token.Length //???? was end = token.index(end, offsetBy: -1)

                    // Check if the substring is good.
                    if String.IsNullOrEmpty currentSubstring then
                        isBad <- true
                        start <- token.Length
                    else
                        subTokens.Add(currentSubstring)
                        start <- fin

                if isBad then 
                    yield unknownToken 
                else 
                    yield! subTokens |]




