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

#r @"..\..\bin\Debug\netcoreapp3.0\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.0\publish\Library.dll"

open System
open System.IO
open Datasets
open DiffSharp

type CycleGANDatasetVariant =
    | Horse2zebra
    member t.url: Uri =
        match t with
        | Horse2zebra -> Uri("https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets")

type CycleGANDataset(trainBatchSize, testBatchSize, ?rootDirPath, ?variant) =

    let variant = defaultArg variant Horse2zebra
    let rootDirPath = 
        match rootDirPath with 
        | None -> CycleGANDataset.downloadIfNotPresent(variant, DatasetUtilities.defaultDirectory </> "CycleGAN")
        | Some p -> p
        
    let trainSamples = 
        Array.zip 
            (CycleGANDataset.loadSamples(rootDirPath </> ("trainA"))) 
            (CycleGANDataset.loadSamples(rootDirPath </> ("trainB")))
        
    let testSamples = 
        Array.zip 
            (CycleGANDataset.loadSamples(rootDirPath </> ("testA"))) 
            (CycleGANDataset.loadSamples(rootDirPath </> ("testB")))

    let training =
        trainSamples
        |> Seq.chunkBySize trainBatchSize
            //entropy: entropy
        |> Seq.map (fun batch -> (dsharp.tensor(Array.map fst batch), dsharp.tensor(Array.map snd batch)))

    let testing = 
        testSamples
        |> Seq.chunkBySize testBatchSize
        |> Seq.map (fun batch -> (dsharp.tensor(Array.map fst batch), dsharp.tensor(Array.map snd batch)))

    member _.TrainingSamples = trainSamples
    member _.TestSamples = testSamples
    member _.TrainingData = training
    member _.TestingData = testing
    
    static member downloadIfNotPresent(variant: CycleGANDatasetVariant, directory: FilePath) =
        let value = variant.ToString().ToLower()
        let rootDirPath = directory </> value

        let directoryExists = File.Exists(rootDirPath)
        if not directoryExists || (Directory.GetFiles(rootDirPath).Length = 0) then 

            DatasetUtilities.downloadResource(
                filename=value+".zip", 
                remoteRoot=variant.url, 
                localStorageDirectory=directory) |> ignore

        rootDirPath


    static member loadSamples(directory: FilePath) : Tensor[] =
        failwith "Tbd"
        //File
        //    .contentsOfDirectory(
        //        at: directory,
        //        includingPropertiesForKeys: [.isDirectoryKey],
        //        options: [.skipsHiddenFiles])
        //    .filter (fun x -> x..pathExtension = "jpg"
        //    .map {
        //        Image(jpeg: $0).tensor / 127.5 - 1.0


