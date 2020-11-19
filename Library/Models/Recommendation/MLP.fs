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

module Models.ImageClassification.MLP

open DiffSharp
open DiffSharp.Model

/// MLP is a multi-layer perceptron and is used as a component of the DLRM model
///
/// Randomly initializes a new multilayer perceptron from the given hyperparameters.
///
/// - Parameter dims: Dims represents the size of the input, hidden layers, and output of the
///   multi-layer perceptron.
/// - Parameter sigmoidLastLayer: if `true`, use a `sigmoid` activation function for the last layer,
///   `relu` otherwise.
type MLP(dims: int[], ?sigmoidLastLayer: bool) =
    inherit Model()
    let sigmoidLastLayer = defaultArg sigmoidLastLayer false
    
    let blocks =
        [| for i in 0..dims.Length-2 do
            if sigmoidLastLayer && i = dims.Length - 2 then
                Linear(inFeatures=dims.[i], outFeatures=dims.[i+1], activation=dsharp.sigmoid)
            else
                Linear(inFeatures=dims.[i], outFeatures=dims.[i+1], activation=dsharp.relu) |]
 
    override _.forward(input) =
        (input, blocks) ||> Array.fold (fun last layer -> layer.forward last) 




