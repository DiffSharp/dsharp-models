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
// "The MNIST database of handwritten digits"
// Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
// http://yann.lecun.com/exdb/mnist/

open DiffSharp

type MNIST(batchSize: int, ?device: Device, ?flattening: bool, ?normalizing, 
        ?localStorageDirectory: FilePath, ?entropy: RandomNumberGenerator) = 

    let localStorageDirectory = defaultArg localStorageDirectory (DatasetUtilities.defaultDirectory </> "KuzushijiMNIST")

    let device = defaultArg device Device.Default
    let flattening = defaultArg flattening false
    let normalizing = defaultArg normalizing false

    // TODO: insert entropy
    let entropy = defaultArg entropy (System.Random())

    /// The training epochs.
    member val training = 
        MNISTHelpers.fetchMNISTDataset(
            localStorageDirectory=localStorageDirectory,
            remoteBaseDirectory="https://storage.googleapis.com/cvdf-datasets/mnist",
            imagesFilename="train-images-idx3-ubyte",
            labelsFilename="train-labels-idx1-ubyte")
        |> Seq.chunkBySize batchSize
        |> Seq.map (fun batch ->  MNISTHelpers.makeMNISTBatch(samples=batch, flattening= flattening, normalizing= normalizing, device=device))

    /// The validation batches.
    member val validation = 
        MNISTHelpers.fetchMNISTDataset(
            localStorageDirectory=localStorageDirectory,
            remoteBaseDirectory="https://storage.googleapis.com/cvdf-datasets/mnist",
            imagesFilename="t10k-images-idx3-ubyte",
            labelsFilename="t10k-labels-idx1-ubyte"
        )
        |> Seq.chunkBySize batchSize
        |> Seq.map (fun chunk -> chunk |> Seq.toArray)
        |> Seq.map (fun batch -> MNISTHelpers.makeMNISTBatch(samples=batch, flattening=flattening, normalizing=normalizing, device=device))
