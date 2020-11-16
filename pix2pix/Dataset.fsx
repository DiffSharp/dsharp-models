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

#r @"..\bin\Debug\netcoreapp3.0\publish\DiffSharp.Core.dll"
#r @"..\bin\Debug\netcoreapp3.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\bin\Debug\netcoreapp3.0\publish\Library.dll"

open System
open Datasets
open DiffSharp

type Pix2PixDatasetVariant =
    | Facades
    member t.url = Uri "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades.zip"

type Pix2PixDataset(?rootDirPath: string,
        ?variant: Pix2PixDatasetVariant, 
        ?trainBatchSize: int,
        ?testBatchSize: int,
        ?entropy: Entropy) =
    let trainBatchSize = defaultArg trainBatchSize 1
    let testBatchSize = defaultArg testBatchSize 1
    let rootDirPath = rootDirPath ?? Pix2PixDataset.downloadIfNotPresent(
        variant: variant ?? .facades,
        Path.Combine(DatasetUtilities.defaultDirectory, "pix2pix"))
    let rootDirURL = Uri(fileURLWithPath= rootDirPath)
        
    let trainSamples = 
        Array.zip
            (Pix2PixDataset.loadSortedSamples(rootDirURL </> "trainA", fileIndexRetriever: "_"))
            (Pix2PixDataset.loadSortedSamples(rootDirURL </> "trainB", fileIndexRetriever: "_"))
        
    let testSamples = 
        Array.zip
           (Pix2PixDataset.loadSortedSamples(rootDirURL </> "testA", fileIndexRetriever: "."))
           (Pix2PixDataset.loadSortedSamples(rootDirURL </> "testB", fileIndexRetriever: "."))

    let training = 
        trainSamples 
        |> Seq.chunkBySize trainBatchSize 
        |> Seq.map (fun batch -> dsharp.tensor(batch |> Array.map (fun x -> x.source)),
                                 dsharp.tensor(batch |> Array.map (fun x -> x.target)))

    let testing = 
        trainSamples 
        |> Seq.chunkBySize testBatchSize 
        |> Seq.map (fun batch -> dsharp.tensor(batch |> Array.map (fun x -> x.source)),
                                 dsharp.tensor(batch |> Array.map (fun x -> x.target)))

    static member downloadIfNotPresent(
            variant: Pix2PixDatasetVariant,
            directory: FilePath) =
        let rootDirPath = directory </> (variant.rawValue).path

        let directoryExists = File.Exists(rootDirPath)
        let contentsOfDir = try? Directory.GetFiles(rootDirPath)
        let directoryEmpty = (contentsOfDir = nil) || (contentsOfDir!.isEmpty)
        guard !directoryExists || directoryEmpty else { return rootDirPath

        let _ = DatasetUtilities.downloadResource(
            filename: variant.rawValue, 
            fileExtension="zip",
            remoteRoot: variant.url.deletingLastPathComponent(), 
            localStorageDirectory: directory)
        print("\(rootDirPath) downloaded.")

        return rootDirPath


    static member loadSortedSamples(
        from directory: Uri, 
        fileIndexRetriever: string
    ) : Tensor =
        return try File
            .contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
            .filter (fun x -> x..pathExtension = "jpg"
            .sorted {
                int($0.lastPathComponent.components(separatedBy: fileIndexRetriever)[0])! <
                int($1.lastPathComponent.components(separatedBy: fileIndexRetriever)[0])!

            .map {
                Image(jpeg: $0).tensor / 127.5 - 1.0






