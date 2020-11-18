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
//
// Adapted from: https://github.com/eaplatanios/nca/blob/master/Sources/NCA/Evaluation.swift

module Models.Evaluation

/// Computes the Matthews correlation coefficient.
///
/// The Matthews correlation coefficient is more informative than other confusion matrix measures
/// (such as F1 score and accuracy) in evaluating binary classification problems, because it takes
/// into account the balance ratios of the four confusion matrix categories (true positives, true
/// negatives, false positives, false negatives).
///
/// - Source: [https://en.wikipedia.org/wiki/Matthews_correlation_coefficient](
///             https://en.wikipedia.org/wiki/Matthews_correlation_coefficient).
let matthewsCorrelationCoefficient(predictions: bool[], groundTruth: bool[]) =
    let mutable tp = 0 // True positives.
    let mutable tn = 0 // True negatives.
    let mutable fp = 0 // False positives.
    let mutable fn = 0 // False negatives.
    for (prediction, truth) in Array.zip predictions groundTruth do
        match (prediction, truth) with
        | (false, false) -> tn <- tn + 1
        | (false, true) -> fn <- fn + 1
        | (true, false) -> fp <- fp + 1
        | (true, true) -> tp <- tp + 1

    let nominator = double(tp * tn - fp * fn)
    let denominator = double((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) |> sqrt
    if denominator <> 0.0 then nominator / denominator else 0.0

