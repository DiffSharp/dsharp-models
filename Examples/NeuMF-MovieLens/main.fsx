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

open Datasets

open RecommendationModels
open DiffSharp

let dataset = MovieLens(trainBatchSize= 1024)
let numUsers = dataset.numUsers
let numItems = dataset.numItems

let size: int[] = [| 16; 32; 16; 8 |]
let regs: double[] = [| 0.0; 0.0; 0.0; 0.0 |]

let model = NeuMF(
    numUsers: numUsers, numItems: numItems, numLatentFeatures: 8, matrixRegularization: 0.0, mlpLayerSizes: size,
    mlpRegularizations: regs)
let optimizer = Adam(model, learningRate=0.001)
let itemCount = Dictionary(
    uniqueKeysWithValues: zip(
        dataset.testUsers, Array.replicate dataset.testUsers.count 0.0))
let testNegSampling = dsharp.zeros([numUsers; numItems])

for element in dataset.testData do
    let rating = element[2]
    if rating > 0 && dataset.item2id.[element[1]] <> nil then
        let uIndex = dataset.user2id[element[0]]!
        let iIndex = dataset.item2id.[element[1]]!
        testNegSampling.[uIndex].[iIndex] = dsharp.tensor(1.0)
        itemCount.[element[0]] = itemCount.[element[0]] + 1.0


print("Dataset acquired.")

print("Starting training..")
let epochCount = 20
for (epoch, epochBatches) in dataset.training.prefix(epochCount).enumerated() do
    let avgLoss: double = 0.0
    model.mode <- Mode.Train
    for batch in epochBatches do
        let userId = batch.first
        let rating = batch.second
        let (loss, grad) = valueWithGradient<| fun model -> 
            let logits = model(userId)
            dsharp.sigmoidCrossEntropy(logits=logits, labels=rating)


        optimizer.update(&model, along=grad)
        avgLoss = avgLoss + loss.toScalar()


    model.mode <- Mode.Eval
    let correct = 0.0
    let count = 0
    for user in dataset.testUsers[0..30] do
        let negativeItem: double[] = [| |]
        let output: double[] = [| |]
        let userIndex = dataset.user2id[user]!
        for item in dataset.items do
            let itemIndex = dataset.item2id.[item]!
            if dataset.trainNegSampling.[userIndex].[itemIndex].toScalar() = 0 then
                let input = Tensor (*<int32>*)(
                    shape=[1, 2], scalars: [int32(userIndex), int32(itemIndex)])
                output.append(model(input).toScalar())
                negativeItem.append(item)


        let itemScore = Dictionary(uniqueKeysWithValues: zip(negativeItem, output))
        let sortedItemScore = itemScore.sorted (fun x -> x.1 > $1.1
        let topK = sortedItemScore.prefix(min(10, int(itemCount.[user]!)))

        for (key, _) in topK do
            if testNegSampling.[userIndex][dataset.item2id.[key]!] = dsharp.tensor(1.0) then
                correct = correct + 1.0

            count = count + 1


    print(
        "Epoch: {epoch}", "Current loss: \(avgLoss/1024.0)", "Validation Accuracy:",
        correct / double(count))


print("Starting testing..")
model.mode <- Mode.Eval
let correct = 0.0
let count = 0
for user in dataset.testUsers do
    let negativeItem: double[] = [| |]
    let output: double[] = [| |]
    let userIndex = dataset.user2id[user]!
    for item in dataset.items do
        let itemIndex = dataset.item2id.[item]!
        if dataset.trainNegSampling.[userIndex][itemIndex].toScalar() = 0 then
            let input = Tensor (*<int32>*)(
                shape=[1, 2], scalars: [int32(userIndex), int32(itemIndex)])
            output.append(model(input).toScalar())
            negativeItem.append(item)



    let itemScore = Dictionary(uniqueKeysWithValues: zip(negativeItem, output))
    let sortedItemScore = itemScore.sorted (fun x -> x.1 > $1.1
    let topK = sortedItemScore.prefix(min(10, int(itemCount.[user]!)))

    print("User:", user, terminator: "\t")
    print("Top K Recommended Items:", terminator: "\t")

    for (key, _) in topK do
        print(key, terminator: "\t")
        if testNegSampling.[userIndex][dataset.item2id.[key]!] = dsharp.tensor(1.0) then
            correct = correct + 1.0

        count = count + 1

    print(terminator: "\n")

print("Test Accuracy:", correct / double(count))
