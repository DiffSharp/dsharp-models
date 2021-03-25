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
                filename=Path.GetFileName(downloadFile),
                remoteRoot= remoteURL, localStorageDirectory=downloadPath,
                extract=false)
            |> ignore

        File.ReadAllText(downloadFile, Encoding.UTF8)

    // Convert Space Separated CSV with no Header
    let dataRecords = 
        data.Split("\n") 
        |> Array.filter(fun line -> line <> "")
        |> Array.map (fun s -> 
            s.Split(" ") 
            |> Array.filter(fun x -> x <> "") 
            |> Array.map float)

    let nRecords = dataRecords.Length
    let nColumns = dataRecords.[0].Length

    let dataFeatures = dataRecords |> Array.map (fun arr -> arr.[0..nColumns - 2])
    let dataLabels = dataRecords |> Array.map (fun arr -> arr.[(nColumns - 1)..])

    // Normalize
    let trainPercentage: double = 0.8

    let nTrainRecords = int(ceil(double(nRecords) * trainPercentage))
    let nTestRecords = nRecords - nTrainRecords

    let xTrainPrelim = dataFeatures.[0..nTrainRecords-1] |> Array.concat
    let xTestPrelim = dataFeatures.[nTrainRecords..]  |> Array.concat
    let yTrainPrelim = dataLabels.[0..nTrainRecords-1]  |> Array.concat
    let yTestPrelim = dataLabels.[nTrainRecords..] |> Array.concat

    let xTrainDeNorm = dsharp.tensor(xTrainPrelim, dtype=Dtype.Float32).view([nTrainRecords; nColumns - 1])
    let xTestDeNorm = dsharp.tensor(xTestPrelim, dtype=Dtype.Float32).view([nTestRecords; nColumns - 1])

    let mean = xTrainDeNorm.mean(dim=0)
    let std = xTrainDeNorm.stddev(dim=0)

    member val numRecords = nRecords
    member val numColumns = nColumns
    member val numTrainRecords = nTrainRecords
    member val numTestRecords = nTestRecords
    member val xTrain = (xTrainDeNorm - mean) / std
    member val xTest = (xTestDeNorm - mean) / std
    member val yTrain = dsharp.tensor(yTrainPrelim, dtype=Dtype.Float32).view([nTrainRecords; 1])
    member val yTest = dsharp.tensor(yTestPrelim, dtype=Dtype.Float32).view([nTestRecords; 1])
