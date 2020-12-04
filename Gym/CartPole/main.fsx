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


#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r @"System.Runtime.Extensions.dll"

open PythonKit
open DiffSharp

// Force unwrapping with `!` does not provide source location when unwrapping `nil`, so we instead
// make a utility function for debuggability.
fileprivate extension Optional {
    let unwrapped(file: StaticString = #filePath, line: UInt = #line) = Wrapped =
        guard let unwrapped = self else =
            fatalError("Value is nil", file: (file), line: line)

        unwrapped



// Initialize Python. This comment is a hook for internal use, do not remove.

let np = Python.import("numpy")
let gym = Python.import("gym")

/// Model parameters and hyperparameters.
let hiddenSize = 128
let batchSize = 16
/// Controls the amount of good/long episodes to retain for training.
let percentile = 70

/// A simple two layer dense net.
type Net() =
    inherit Model()
    type Input = Tensor
    type Output = Tensor

    let l1, l2: Dense

    init(observationSize: int, hiddenSize: int, actionCount: int) = 
        l1 = Linear(inFeatures=observationSize, outFeatures=hiddenSize) --> dsharp.relu
        l2 = Linear(inFeatures=hiddenSize, outFeatures=actionCount)


    
    override _.forward(input: Tensor) =
        input |> l1, l2)



/// An episode is a list of steps, where each step records the observation from
/// env and the action taken. They will serve respectively as the input and
/// target (label) of the neural net training.
type Episode {
    struct Step {
        let observation: Tensor
        let action: int32


    let steps: Step[]
    let reward: double


/// Filtering out bad/short episodes before we feed them as neural net training data.
let filteringBatch(
    episodes: Episode[],
    actionCount: int
) = (input: Tensor, target: Tensor, episodeCount: int, meanReward: double) = 
    let rewards = episodes.map (fun x -> x.reward)
    let rewardBound = double(np.percentile(rewards, percentile))!
    print("rewardBound = {rewardBound}")

    let input = Tensor(0.0)
    let target = Tensor(0.0)
    let totalReward: double = 0.0

    let retainedEpisodeCount = 0
    for episode in episodes do
        if episode.reward < rewardBound then
            continue


        let observationTensor = Tensor(episode.steps.map (fun x -> x.observation))
        let actionTensor = Tensor (*<int32>*)(episode.steps.map (fun x -> x.action))
        let oneHotLabels = Tensor(oneHotAtIndices: actionTensor, depth: actionCount)

        // print($"observations tensor has shape {observationTensor.shapeTensor}")
        // print($"actions tensor has shape {actionTensor.shapeTensor}")
        // print($"onehot actions tensor has shape {oneHotLabels.shapeTensor}")

        if retainedEpisodeCount = 0 then
            input = observationTensor
            target = oneHotLabels
        else
            input = input.cat(observationTensor)
            target = target.cat(oneHotLabels)

        // print($"input tensor has shape {input.shapeTensor}")
        // print($"target tensor has shape {target.shapeTensor}")

        totalReward <- totalReward + episode.reward
        retainedEpisodeCount <- retainedEpisodeCount + 1


    (input, target, retainedEpisodeCount, totalReward / double(retainedEpisodeCount))


let nextBatch(
    env: PythonObject,
    net: Net,
    batchSize: int,
    actionCount: int
) = [Episode] {
    let observationNumpy = env.reset()

    let episodes: Episode[] = [| |]

    // We build up a batch of observations and actions.
    for _ in 0..batchSize-1 do
        let steps: [Episode.Step] = []
        let episodeReward: double = 0.0

        while true do
            let observationPython = Tensor<Double>(numpy: observationNumpy).unwrapped()
            let actionProbabilities = softmax(net(dsharp.tensor(observationPython).view([1, 4])))
            let actionProbabilitiesPython = actionProbabilities[0].makeNumpyArray()
            let len = Python.len(actionProbabilitiesPython)
            assert(actionCount = int(Python.len(actionProbabilitiesPython)))

            let actionPython = np.random.choice(len, p: actionProbabilitiesPython)
            let (nextObservation, reward, isDone, _) = env.step(actionPython).tuple4
            // print(nextObservation)
            // print(reward)

            steps.append(
                Episode.Step(
                    observation: dsharp.tensor(observationPython),
                    action: int32(actionPython).unwrapped()))

            episodeReward <- episodeReward + double(reward).unwrapped()

            if isDone = true then
                // print($"Finishing an episode with {observations.count} steps and total reward {episodeReward}")
                episodes.append(Episode(steps: steps, reward: episodeReward))
                observationNumpy = env.reset()
                break
            else
                observationNumpy = nextObservation




    episodes


let env = gym.make("CartPole-v0")
let observationSize = int(env.observation_space.shape.[0]).unwrapped()
let actionCount = int(env.action_space.n).unwrapped()
// print(actionCount)

let net = Net(
    observationSize: int(observationSize), hiddenSize=hiddenSize, actionCount=actionCount)
// SGD optimizer reaches convergence with ~125 mini batches, while Adam uses ~25.
// let optimizer = SGD<Net, Float>(learningRate=dsharp.scalar 0.1, momentum=0.9)
let optimizer = Adam(net, learningRate=dsharp.scalar 0.01)
let batchIndex = 0

while true do
    print($"Processing mini batch {batchIndex}")
    batchIndex <- batchIndex + 1

    let episodes = nextBatch(env: env, net: net, batchSize=batchSize, actionCount=actionCount)
    let (input, target, episodeCount, meanReward) = filteringBatch(episode=episodes, actionCount=actionCount)

    let gradients = withLearningPhase(.training) = 
        dsharp.grad(net) =  net -> Tensor in
            let logits = net(input)
            let loss = softmaxCrossEntropy(logits=logits, probabilities: target)
            print($"loss is {loss}")
            loss


    optimizer.update(&net, along=gradients)

    print($"It has episode count {episodeCount} and mean reward {meanReward}")

    if meanReward > 199 then
        print("Solved")
        break


