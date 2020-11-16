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

// Original source:
// "The Boston Housing dataset"
// Harrison, D. and Rubinfeld, D.L..
// https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

namespace Datasets

open System
open System.IO
open System.Text
open DiffSharp

type BostonHousing() =
    let data = 
        let remoteURL = Uri("https://storage.googleapis.com/s4tf-hosted-binaries/datasets/BostonHousing/")
        let downloadPath = Path.Combine(DatasetUtilities.defaultDirectory, "BostonHousing")
        let downloadFile = Path.Combine(downloadPath,"housing.data")

        if not (File.Exists(downloadPath)) || Directory.GetFiles(downloadPath) |> Array.isEmpty  then
            DatasetUtilities.downloadResource(
                filename=downloadFile,
                remoteRoot= remoteURL, localStorageDirectory=downloadPath,
                extract=false)
            |> ignore

        File.ReadAllText(downloadFile, Encoding.UTF8)

    // Convert Space Separated CSV with no Header
    let dataRecords = data.Split("\n") |> Array.map (fun s -> s.Split(" ") |> Array.map float)

    let numRecords = dataRecords.Length
    let numColumns = dataRecords.[0].Length

    let dataFeatures = dataRecords |> Array.map (fun arr -> arr.[0..numColumns - 2])
    let dataLabels = dataRecords |> Array.map (fun arr -> arr.[(numColumns - 1)..])

    // Normalize
    let trainPercentage: double = 0.8

    let numTrainRecords = int(ceil(double(numRecords) * trainPercentage))
    let numTestRecords = numRecords - numTrainRecords

    let xTrainPrelim = dataFeatures.[0..numTrainRecords-1] |> Array.concat
    let xTestPrelim = dataFeatures.[numTrainRecords..]  |> Array.concat
    let yTrainPrelim = dataLabels.[0..numTrainRecords-1]  |> Array.concat
    let yTestPrelim = dataLabels.[numTrainRecords..] |> Array.concat

    let xTrainDeNorm = dsharp.tensor(xTrainPrelim, dtype=Dtype.Float32).view([numTrainRecords; numColumns - 1])
    let xTestDeNorm = dsharp.tensor(xTestPrelim, dtype=Dtype.Float32).view([numTestRecords; numColumns - 1])

    let mean = xTrainDeNorm.mean(dim=0)
    let std = xTrainDeNorm.stddev(dim=0)

    member val xTrain = (xTrainDeNorm - mean) / std
    member val xTest = (xTestDeNorm - mean) / std
    member val yTrain = dsharp.tensor(yTrainPrelim, dtype=Dtype.Float32).view([numTrainRecords; 1])
    member val yTest = dsharp.tensor(yTestPrelim, dtype=Dtype.Float32).view([numTestRecords; 1])
