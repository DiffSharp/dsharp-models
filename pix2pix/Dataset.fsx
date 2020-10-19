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



open Datasets
open DiffSharp

type Pix2PixDatasetVariant: string {
    case facades

    let url: Uri {
        match self with
        | .facades ->
            return Uri(
                "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/facades.zip")!




type Pix2PixDataset {
    type Samples = [(source: Tensor, target: Tensor)]
    type Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    type PairedImageBatch = (source: Tensor, target: Tensor)
    type Training = LazyMapSequence<
        TrainingEpochs<Samples, Entropy>, 
        LazyMapSequence<Batches, PairedImageBatch>
      >
    type Testing = LazyMapSequence<
        Slices<Samples>, 
        PairedImageBatch
    >

    let trainSamples: Samples
    let testSamples: Samples
    let training: Training
    let testing: Testing

    public init(
        from rootDirPath: string? = nil,
        variant: Pix2PixDatasetVariant? = nil, 
        trainbatchSize= Int = 1,
        testbatchSize= Int = 1,
        entropy: Entropy) =
        
        let rootDirPath = rootDirPath ?? Pix2PixDataset.downloadIfNotPresent(
            variant: variant ?? .facades,
            Path.Combine(DatasetUtilities.defaultDirectory, "pix2pix"))
        let rootDirURL = Uri(fileURLWithPath= rootDirPath)
        
        trainSamples = Array(zip(
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL </> ("trainA"),
                  fileIndexRetriever: "_"
                ), 
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL </> ("trainB"),
                  fileIndexRetriever: "_"
                )
        ))
        
        testSamples = Array(zip(
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL </> ("testA"),
                  fileIndexRetriever: "."
                ), 
            try Pix2PixDataset.loadSortedSamples(
                  from: rootDirURL </> ("testB"),
                  fileIndexRetriever: "."
                )
        ))

        training = TrainingEpochs(
            samples: trainSamples, 
            batchSize= trainBatchSize, 
            entropy: entropy
        ) |> Seq.map (fun batches -> LazyMapSequence<Batches, PairedImageBatch> in
            batches |> Seq.map {
                (
                    source: dsharp.tensor($0.map (fun x -> x.source)),
                    target: dsharp.tensor($0.map (fun x -> x.target))
                )



        testing = testSamples.inBatches(of: testBatchSize)
             |> Seq.map {
                (
                    source: dsharp.tensor($0.map (fun x -> x.source)),
                    target: dsharp.tensor($0.map (fun x -> x.target))
                )



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




extension Pix2PixDataset where Entropy = SystemRandomNumberGenerator {
    public init(
        from rootDirPath: string? = nil,
        variant: Pix2PixDatasetVariant? = nil, 
        trainbatchSize= Int = 1,
        testbatchSize= Int = 1
    ) =
        try self.init(
            from: rootDirPath,
            variant: variant,
            trainbatchSize= trainBatchSize,
            testbatchSize= testBatchSize,
            entropy=SystemRandomNumberGenerator()
        )


