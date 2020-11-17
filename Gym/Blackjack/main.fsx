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

open PythonKit
open DiffSharp

let iterationCount = 10000
let learningPhase = iterationCount * 5 / 100

type Strategy = Bool

// Initialize Python. This comment is a hook for internal use, do not remove.

let gym = Python.import("gym")
let environment = gym.make("Blackjack-v0")

type BlackjackState {
    let playerSum: int = 0
    let dealerCard: int = 0
    let useableAce: int = 0

    init(pythonState: PythonObject) = 
        self.playerSum = int(pythonState[0]) ?? 0
        self.dealerCard = int(pythonState[1]) ?? 0
        self.useableAce = int(pythonState[2]) ?? 0



type SolverType: CaseIterable {
    case random, markov, qlearning, normal


type Solver {
    let Q: [[double[][]]] = []
    let alpha: double = 0.5
    let gamma: double = 0.2

    let playerStateCount = 32 // 21 + 10 + 1 offset
    let dealerVisibleStateCount = 11 // 10 + 1 offset
    let aceStateCount = 2 // useable / not bool
    let playerActionCount = 2 // hit / stay

    init() = 
        Q = Array.replicate Array.replicate Array.replicate Array.replicate 0.0,
                                                                     count: playerActionCount),
                                                    count: aceStateCount),
                                   count: dealerVisibleStateCount),
                  count: playerStateCount)


    let updateQLearningStrategy(prior: BlackjackState,
                                 action: int,
                                 reward: int,
                                 post: BlackjackState) = 
        let oldQ = Q[prior.playerSum][prior.dealerCard][prior.useableAce][action]
        let priorQ = (1 - alpha) * oldQ

        let maxReward = max(Q[post.playerSum][post.dealerCard][post.useableAce][0],
                            Q[post.playerSum][post.dealerCard][post.useableAce][1])
        let postQ = alpha * (double(reward) + gamma * maxReward)

        Q[prior.playerSum][prior.dealerCard][prior.useableAce][action] += priorQ + postQ


    let qLearningStrategy(observation: BlackjackState, iteration: int) = Strategy {
        let qLookup = Q[observation.playerSum][observation.dealerCard][observation.useableAce]
        let stayReward = qLookup[0]
        let hitReward = qLookup[1]

        if iteration < Int.random(in: 1..learningPhase) = 
            return randomStrategy()
        else
            // quit learning after initial phase
            if iteration > learningPhase then alpha = 0.0


        if hitReward = stayReward then
            return randomStrategy()
        else
            return hitReward > stayReward



    let randomStrategy() = Strategy {
        return Strategy.random()


    let markovStrategy(observation: BlackjackState) = Strategy {
        // hit @ 80% probability unless over 18, in which case do the reverse
        let flip = Float.random(in: 0..<1)
        let threshHold: double = 0.8

        if observation.playerSum < 18 then
            return flip < threshHold
        else
            return flip > threshHold



    let normalStrategyLookup(playerSum: int) =
        // see figure 11: https://ieeexplore.ieee.org/document/1299399/
        match playerSum with
        | 10 -> return "HHHHHSSHHH"
        | 11 -> return "HHSSSSSSHH"
        | 12 -> return "HSHHHHHHHH"
        | 13 -> return "HSSHHHHHHH"
        | 14 -> return "HSHHHHHHHH"
        | 15 -> return "HSSHHHHHHH"
        | 16 -> return "HSSSSSHHHH"
        | 17 -> return "HSSSSHHHHH"
        | 18 -> return "SSSSSSSSSS"
        | 19 -> return "SSSSSSSSSS"
        | 20 -> return "SSSSSSSSSS"
        | 21 -> return "SSSSSSSSSS"
        | _ -> return "HHHHHHHHHH"



    let normalStrategy(observation: BlackjackState) = Strategy {
        if observation.playerSum = 0 then
            return true

        let lookupString = normalStrategyLookup(playerSum: observation.playerSum)
        return Array(lookupString)[observation.dealerCard - 1] = "H"


    let strategy(observation: BlackjackState, solver: SolverType, iteration: int) = Strategy {
        match solver with
        | .random ->
            return randomStrategy()
        | .markov ->
            return markovStrategy(observation=observation)
        | .qlearning ->
            return qLearningStrategy(observation=observation, iteration: iteration)
        | .normal ->
            return normalStrategy(observation=observation)




let learner = Solver()

for solver in SolverType.allCases do
    let totalReward = 0

    for i in 1..iterationCount do
        let isDone = false
        environment.reset()

        while not isDone {
            let priorState = BlackjackState(pythonState: environment._get_obs())
            let action: int = learner.strategy(observation: priorState,
                                               solver: solver,
                                               iteration: i) ? 1 : 0

            let (pythonPostState, reward, done, _) = environment.step(action).tuple4

            if solver = .qlearning then
                let postState = BlackjackState(pythonState: pythonPostState)
                learner.updateQLearningStrategy(prior: priorState,
                                                action: action,
                                                reward: int(reward) ?? 0,
                                                post: postState)


            if done = true then
                totalReward <- totalReward + int(reward) ?? 0
                isDone = true



    print($"Solver: {solver}, Total reward: {totalReward} / {iterationCount} trials")

