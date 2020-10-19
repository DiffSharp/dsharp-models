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

namespace Datasets

// Original source:
// "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms"
// Han Xiao and Kashif Rasul and Roland Vollgraf
// https://arxiv.org/abs/1708.07747

open DiffSharp
open System

// /// Type of the collection of non-collated batches.
// type Batches = Slices<Sampling<(byte[] * int32)[], ArraySlice<Int>>>
/// The type of the training data, represented as a sequence of epochs, which
/// are collection of batches.
type Training = seq<{| data: byte[]; label: int32 |}[]> // ,seq<Batches, LabeledImage>>
/// The type of the validation data, represented as a collection of batches.
type Validation = seq<{| data: byte[]; label: int32 |}[]> //, LabeledImage>

/// Creates an instance with `batchSize` on `device`.
///
/// - Parameters:
///   - entropy: a source of randomness used to shuffle sample ordering.  It  
///     will be stored in `self`, so if it is only pseudorandom and has value 
///     semantics, the sequence of epochs is deterministic and not dependent 
///     on other operations.
///   - flattening: flattens the data to be a 2d-tensor iff `true. The default value
///     is `false`.
///   - normalizing: normalizes the batches to have values from -1.0 to 1.0 iff `true`.
///     The default value is `false`.
///   - localStorageDirectory: the directory in which the dataset is stored.
type FashionMNIST(batchSize: int, device: Device, ?flattening: bool, ?normalizing, 
                  ?localStorageDirectory: FilePath, ?entropy: RandomNumberGenerator) = 
    let localStorageDirectory = defaultArg localStorageDirectory (DatasetUtilities.defaultDirectory </> "FashionMNIST")
    let flattening = defaultArg flattening false
    let normalizing = defaultArg normalizing false

    // TODO: insert entropy
    let entropy = defaultArg entropy (System.Random())

    /// The training epochs.
    member val training = 
        MNISTHelpers.fetchMNISTDataset(
            localStorageDirectory=localStorageDirectory,
            remoteBaseDirectory="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
            imagesFilename="train-images-idx3-ubyte",
            labelsFilename="train-labels-idx1-ubyte")
        |> Seq.chunkBySize batchSize
        |> Seq.map (fun batch ->  MNISTHelpers.makeMNISTBatch(samples=batch, flattening= flattening, normalizing= normalizing, device=device))

    /// The validation batches.
    member val validation = 
        MNISTHelpers.fetchMNISTDataset(
            localStorageDirectory=localStorageDirectory,
            remoteBaseDirectory="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
            imagesFilename="t10k-images-idx3-ubyte",
            labelsFilename="t10k-labels-idx1-ubyte"
        )
        |> Seq.chunkBySize batchSize
        |> Seq.map (fun chunk -> chunk |> Seq.toArray)
        |> Seq.map (fun batch -> MNISTHelpers.makeMNISTBatch(samples=batch, flattening=flattening, normalizing=normalizing, device=device))

