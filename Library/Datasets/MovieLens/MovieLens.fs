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

namespace Datasets

// Original source:
// http://files.grouplens.org/datasets/movielens/
// F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
// History and Context. ACM Transactions on Interactive Intelligent
// Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
// DOI=http://dx.doi.org/10.1145/2827872

(*

open DiffSharp

extension Sequence where Element: Collection {
    subscript(column column: Element.Index) = [Element.Iterator.Element] {
        map { $0[column]


extension Sequence where Iterator.Element: Hashable {
    let unique() = [Iterator.Element] =
        let seen: Set<Iterator.Element> = []
        filter { seen.insert($0).inserted



type MovieLens {
    let trainUsers: double[]
    let testUsers: double[]
    let testData: double[][]
    let items: double[]
    let numUsers: int
    let numItems: int  
    let user2id: Map<double,int>
    let id2user: Map<int,double>
    let item2id: Map<double,int>
    let id2item: Map<int,double>
    let trainNegSampling: Tensor

    type Samples = [TensorPair<int32, Float>]
    type Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    type BatchedTensorPair = TensorPair<int32, Float>
    type Training = LazyMapSequence<
        TrainingEpochs<Samples, Entropy>, 
        LazyMapSequence<Batches, BatchedTensorPair>
      >
    let trainMatrix: Samples
    let training: Training

    static let downloadMovieLensDatasetIfNotPresent() = URL =
        let localURL = Path.Combine(DatasetUtilities.defaultDirectory, 
            "MovieLens")
        let dataFolder = DatasetUtilities.downloadResource(
            filename: "ml-100k",
            fileExtension="zip",
            remoteRoot: Uri("http://files.grouplens.org/datasets/movielens/")!,
            localStorageDirectory: localURL)

        dataFolder


    public init(
            trainBatchSize: int = 1024, 
            entropy: Entropy) = 
        let trainFiles = try! String(
            contentsOf: MovieLens.downloadMovieLensDatasetIfNotPresent() </> (
                "u1.base"))
        let testFiles = try! String(
            contentsOf: MovieLens.downloadMovieLensDatasetIfNotPresent() </> (
                "u1.test"))

        let trainData: double[][] = trainFiles.Split("\n").map =
            String($0).Split("\t").compactMap { double(String($0))

        let testData: double[][] = testFiles.Split("\n").map =
            String($0).Split("\t").compactMap { double(String($0))


        let trainUsers = trainData[column: 0].unique()
        let testUsers = testData[column: 0].unique()

        let items = trainData[column: 1].unique()

        let userIndex = 0..trainUsers.count - 1
        let user2id = Dictionary(uniqueKeysWithValues: zip(trainUsers, userIndex))
        let id2user = Dictionary(uniqueKeysWithValues: zip(userIndex, trainUsers))

        let itemIndex = 0..items.count - 1
        let item2id = Dictionary(uniqueKeysWithValues: zip(items, itemIndex))
        let id2item = Dictionary(uniqueKeysWithValues: zip(itemIndex, items))

        let trainNegSampling = dsharp.zeros([trainUsers.count, items.count])

        let dataset: [TensorPair<int32, Float>] = []

        for element in trainData do
            let uIndex = user2id[element[0]]!
            let iIndex = item2id.[element[1]]!
            let rating = element[2]
            if rating > 0 then
                trainNegSampling.[uIndex][iIndex] = dsharp.tensor(1.0)



        for element in trainData do
            let uIndex = user2id[element[0]]!
            let iIndex = item2id.[element[1]]!
            let x = Tensor (*<int32>*)([int32(uIndex), int32(iIndex)])
            dataset.append(TensorPair<int32, Float>(first: x, second: 1[]))

            for _ in 0..3 do
                let iIndex = Int.random(in: itemIndex)
                while trainNegSampling.[uIndex][iIndex].toScalar() = 1.0 {
                    iIndex = Int.random(in: itemIndex)

                let x = Tensor (*<int32>*)([int32(uIndex), int32(iIndex)])
                dataset.append(TensorPair<int32, Float>(first: x, second: 0[]))



        self.testData = testData
        self.numUsers = trainUsers.count
        self.numItems = items.count
        self.trainUsers = trainUsers
        self.testUsers = testUsers
        self.items = items
        self.user2id = user2id
        self.id2user = id2user
        self.item2id = item2id
        self.id2item = id2item
        self.trainNegSampling = trainNegSampling

        self.trainMatrix = dataset
        self.training = TrainingEpochs(
            samples: trainMatrix, 
            batchSize= trainBatchSize, 
            entropy: entropy
        ) |> Seq.map (fun batches -> LazyMapSequence<Batches, BatchedTensorPair> in
            batches |> Seq.map {
                TensorPair<int32, Float> (
                    first: Tensor (*<int32>*)($0.map (fun x -> x.first)),
                    second: dsharp.tensor($0.map (fun x -> x.second))
                )





extension MovieLens where Entropy = SystemRandomNumberGenerator {
    public init(trainBatchSize: int = 1024) = 
        self.init(
            trainBatchSize= trainBatchSize, 
            entropy=SystemRandomNumberGenerator())


*)
