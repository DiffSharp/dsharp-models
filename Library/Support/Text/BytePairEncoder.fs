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

namespace Support.Text

open DiffSharp
open Support
open System
open System.IO
open System.Text
open System.Text.RegularExpressions

type Variant =
    /// Default variant.
    /// - Source: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](
    ///             https://arxiv.org/pdf/1907.11692.pdf).
    | Roberta
    /// - Source: [Language Models are Unsupervised Multitask Learners](
    ///             https://cdn.openai.com/better-language-models/
    ///             language_models_are_unsupervised_multitask_learners.pdf).
    | Gpt2

// TODO: Find a nice way to support caching.
/// A cache used to store encoded tokens and thus speed up encoding.
//  let cache: [String: string[]]
type BytePairEncoder(vocabulary: Vocabulary, mergePairs: ((string * string) * int)[], ?useCache: bool) = 
    let mergePairs = Map.ofArray mergePairs
    let useCache = defaultArg useCache true
    let reversedMergePairs = Map.ofSeq (mergePairs |> Seq.map (fun kvp -> fst kvp.Key + snd kvp.Key, kvp.Key))
 
    static let defaultGlossary = [|
        "e.g"; "i.e"; "&amp;"; "&#124;"; "&lt;"; "&gt;"; "&apos;"; "&quot;"; "&#91;"; "&#93;";
      |]

    static let defaultGlossaryRegex: Regex = 
        let escapedGlossary = defaultGlossary |> Array.map (fun x -> $"\\Q{x}\\E") |> String.concat "|"
        Regex("(?:{escapedGlossary})|(?!{escapedGlossary})")

    /// Regular expression matching the OpenAI GPT-2 implementation.
    static let gpt2Glossary = [|
        "'s"; "'t"; "'re"; "'ve"; "'m"; "'ll"; "'d"; " ?\\p{L+"; " ?\\p{N+";
        " ?[^\\s\\p{L\\p{N]+"; "\\s+(?!\\S)"; "\\s+";
      |]

    static let gpt2GlossaryRegex: Regex = 
        let escapedGlossary = gpt2Glossary |> String.concat "|"
        Regex("(?:{escapedGlossary})")

    // TODO: Add documentation.
    static let bytesToUnicode = 
        let bytes = ResizeArray ([| yield! [| 33uy..126uy |]; yield! [| 161uy..172uy |]; yield!  [| 174uy..255uy |] |])
        let characters = bytes.ToArray() |> Array.map uint32 |> ResizeArray
        let mutable offset = 0u
        for byte in 0uy .. 255uy do 
            if not (bytes.Contains(byte)) then
                bytes.Add(byte)
                characters.Add(offset + 256u)
                offset <- offset + 1u

        Map.ofArray (Array.zip (bytes.ToArray()) (characters.ToArray() |> Array.map (fun x -> char(x))))

    // The inverse of bytesToUnicode.
    static let unicodeToBytes = Map.ofSeq (bytesToUnicode |> Seq.map (fun (KeyValue(a,b)) -> (b,a)))

    /// Recursively splits `token` into smaller units (by reversing BPE merges) until all units
    /// are either in the provided vocabulary, or cannot be split further.
    ///
    /// - Parameters:
    ///   - token: Token that needs to be split.
    let rec splitRecursively(token: string) : string[] =
        if reversedMergePairs.ContainsKey token then
            let pair = reversedMergePairs.[token]
            let leftParts = if vocabulary.contains(fst pair) then [| fst pair |] else splitRecursively(fst pair)
            let rightParts = if vocabulary.contains(snd pair) then [| snd pair |] else splitRecursively(snd pair)
            Array.append leftParts rightParts
        else 
            [| token |]

    /// Uses the given regex to split a token into individual glossary terms.
    ///
    /// - Parameters:
    ///   - token: Full text.
    ///   - glossaryRegex: Regular expression for segmenting the given token.
    ///   - variant=The type of model (| _ -> .roberta).
    /// - Returns: Array of substrings that match the given regex.
    let splittingWithDelimiters(token: string, glossaryRegex: Regex, variant: Variant) : string[] =

        let keepEmpty = false

        let matches = failwith "Tbd" // glossaryRegex.matches(in: token, range: NSRange(token.startIndex.., in: token))
        let parts = ResizeArray<string>()
        parts.Capacity <- token.Length
        match variant with
        | Gpt2 ->
            for m in matches do
                failwith "TBD"
                //let start = token.index(token.startIndex, offsetBy: m.range.lowerBound, limitedBy: token.endIndex)
                //let fin = token.index(token.startIndex, offsetBy: m.range.upperBound, limitedBy: token.endIndex) 
                //if ok then
                //  parts.Add(String(token.[start..fin-1]))

        | Roberta ->
            failwith "TBD"
            //let mutable lastEnd = token.startIndex
            //for m in matches do
            //    let start = token.index(token.startIndex, offsetBy: m.range.lowerBound)
            //    if lastEnd <> start then parts.Add(String(token.[lastEnd..<start]))
            //    lastEnd <- token.index(token.startIndex, offsetBy: m.range.upperBound)

            //if lastEnd <> token.endIndex then
            //    parts.Add(String(token.[lastEnd..]))

        parts.ToArray()

    /// Replaces all occurrences of the provided symbol pair in `token` with the joined symbol.
    ///
    /// - Parameters:
    ///   - pair: Symbol pair to replace in `token`.
    ///   - token: Token as a sequence of symbols.
    /// - Returns: New token with the provided pair replaced for the joined symbol.
    static let replacePair(pair: (string * string), tokenParts: string[]) : string[] =
        let newTokenParts = ResizeArray<string>()
        newTokenParts.Capacity <- tokenParts.Length
        let mutable j = 0
        while j < tokenParts.Length - 1 do
            let part1 = tokenParts.[j]
            let part2 = tokenParts.[j + 1]
            if part1 = fst pair && part2 = snd pair then
                let joinedPair = part1 + part2
                newTokenParts.Add(joinedPair)
                j <- j + 2
            else
                newTokenParts.Add(tokenParts.[j])
                j <- j + 1

        if j = tokenParts.Length - 1 then
            newTokenParts.Add(tokenParts.[j])

        newTokenParts.ToArray()

    static let encodedToken(token: string) : string =
        failwith "TBD"
        //String(String.UnicodeScalarView(token.utf8.map { BytePairEncoder.bytesToUnicode[$0]!))

    /// Decodes the provided BPE-coded token to a sequence of tokens.
    ///
    /// - Parameters:
    ///   - token: BPE-coded token to decode.
    /// - Returns: string containing the decoded tokens.
    static member decode(token: string) =
        let buffer = ResizeArray<byte>()

        failwith "TBD"
        //for scalar in token.unicodeScalars do
        //    buffer.Add(BytePairEncoder.unicodeToBytes.[scalar])

        //try 
        //    String(buffer.ToArray())
        //with _ -> 
        //    "\u{FFFD}"

    static member FromVocabularyFile(vocabularyFile: FilePath, mergesFile: FilePath, ?encoding: Encoding, ?useCache: bool) =
        let vocabulary: Vocabulary = Vocabulary.FromJsonFile(vocabularyFile)

        let lines =
            File.ReadAllLines(mergesFile, defaultArg encoding Encoding.UTF8)
            |> Array.tail

        let pairs =
            [| for (index, line) in Seq.indexed lines do
                let tokens = line.Split(" ")
                if tokens.Length <= 2 then 
                    ((tokens.[0], tokens.[1]), index) |]
        BytePairEncoder(vocabulary, pairs, ?useCache=useCache)

    /// Encodes the provided token to a sequence of BPE-coded tokens.
    ///
    /// - Parameters:
    ///   - token: Token to encode.
    ///   - variant=Type of model (| _ -> .roberta).
    /// - Returns: Array containing the BPE-coded tokens.
    member _.encode(token: string, ?variant: Variant) : string[] =
        let variant = defaultArg variant Roberta
        // if let cached = cache[token] then return cached
        // let token = " " + token
         
        let parts = 
            match variant with 
            | Gpt2 ->
                // Split into parts before encoding.
                let unencodedTokens = splittingWithDelimiters(token, gpt2GlossaryRegex, Gpt2)
                // Encode each token.
                let tokens = unencodedTokens |> Array.map encodedToken
                // Separate each character. 
                for token in tokens do
                    for i in 0..token.Length-1 do
                        failwith "TBD"
                        //let index = token.index(token.startIndex, offsetBy: i)
                        //parts.Add(String(token.[index]))

                //if parts.Count < 2 then parts
                [| |]
            | Roberta ->
                // Encode before splitting into parts.
                let encodedToken = encodedToken(token)
                splittingWithDelimiters(encodedToken, defaultGlossaryRegex,Roberta)
                //if parts.Count < 2 then parts

        // Create pairs of parts.
        let mutable pairs = [|0 .. parts.Length - 2 |] |> Array.map (fun index -> (parts.[index], parts.[index + 1]))
        let mutable fin = false
        while not fin && pairs.Length > 0 do
            let pair = pairs |> Array.minBy (fun x -> defaultArg (mergePairs.TryFind x) Int32.MaxValue)
            if not (mergePairs.ContainsKey(pair)) then  fin <- true else
            let parts = replacePair(pair, parts)
            if parts.Length < 2 then fin <- true else
            pairs <- [| 0..parts.Length - 2 |] |> Array.map (fun  index -> (parts.[index], parts.[index + 1]))

        // Check if the new word parts are in the vocabulary, and backtrack if necessary.
        let encoded = 
            parts |> Array.collect (fun part ->
                if vocabulary.contains(part) then [| part |] else
                splitRecursively(part))

        // Update the cache and return.
        // if useCache then cache[token] = encoded
        encoded

