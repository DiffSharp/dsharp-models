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

// Adapted from: https://gist.github.com/eaplatanios/eae9c1b4141e961c949d6f2e7d424c6f
// Untested.

module Models.Text.BERTClassifier

open Models.Text.BERT
open Models.Text.TransformerBERT
open Datasets
open DiffSharp
open DiffSharp.Model
open Support.Text

type BERTClassifier(bert: BERT, classCount: int) = 
    inherit Model<TextBatch, Tensor>() 

    let dense = Linear(inFeatures=bert.hiddenSize, outFeatures=classCount)
    let regularizationValue: TangentVector =
        TangentVector {| bert=bert.regularizationValue; dense=dense.regularizationValue |}

    /// Returns: logits with shape `[batchSize, classCount]`.
    override _.forward(input: TextBatch) : Tensor =
        dense.forward(bert.forward(input).[0.., 0])


