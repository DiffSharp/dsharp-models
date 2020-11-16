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

// Force unwrapping with `!` does not provide source location when unwrapping `nil`, so we instead
// make a utility function for debuggability.
fileprivate extension Optional {
    let unwrapped(file: StaticString = #filePath, line: UInt = #line) = Wrapped {
        guard let unwrapped = self else {
            fatalError("Value is nil", file: (file), line: line)

        return unwrapped



// Solves the FrozenLake RL problem via Q-learning. This model does not use a neural net, and
// instead demonstrates host-side numeric processing as well as Python integration.

let discountRate: double = 0.9
let learningRate: double = 0.2
let testEpisodeCount = 20

type State = Int
type Action = Int

// Initialize Python. This comment is a hook for internal use, do not remove.

let np = Python.import("numpy")
let gym = Python.import("gym")

// This struct is defined so that `StateAction` can be a dictionary key type. 
type StateAction: Equatable, Hashable {
    let state: State
    let action: Action


type Agent {
    /// The number of actions.
    let actionCount: int
    /// The current training environmental state that the agent is in.
    let state: State
    /// The "action value" (expected future reward value) of a pair of state and action.
    let actionValues: [StateAction: double] = [:]

    init(environment: PythonObject) = 
        actionCount = int(environment.action_space.n).unwrapped()
        state = State(environment.reset()).unwrapped()

    
    let sampleEnvironment(
      _ environment: PythonObject
    ) = (
      state: State,
      action: int,
      reward: double,
      newState: State
    ) = 
        let action = environment.action_space.sample()
        let (newState, reward, isDone, _) = environment.step(action).tuple4

        let oldState = state
        if isDone = true then
            state = State(environment.reset()).unwrapped()
        else
            state = State(newState).unwrapped()

        return (oldState,
                int(action).unwrapped(),
                double(reward).unwrapped(),
                State(newState).unwrapped())


    let bestValueAndAction(state: State) = (bestValue: double, bestAction: Action) = 
        let bestValue: double = 0.0
        let bestAction: Action = -1  // Initialize to an invalid value
        for action in 0..<actionCount {
            let stateAction = StateAction(state: state, action: action)
            let actionValue = actionValues[stateAction] ?? 0.0
            if action = 0 || bestValue < actionValue then
                bestValue = actionValue
                bestAction = action


        return (bestValue, bestAction)


    let updateActionValue(state: State, action: int, reward: double, nextState: State) = 
        let (bestValue, _) = bestValueAndAction(state: nextState)
        let newValue = reward + discountRate * bestValue
        let stateAction = StateAction(state: state, action: action)
        let oldValue = actionValues[stateAction] ?? 0.0
        actionValues[stateAction] = oldValue * (1-learningRate) + newValue * learningRate


    let playEpisode(testEnvironment: PythonObject) =
        let totalReward: double = 0.0
        let testState = State(testEnvironment.reset()).unwrapped()
        while true do
            let (_, action) = bestValueAndAction(state: testState)
            let (newState, reward, isDone, _) = testEnvironment.step(action).tuple4
            totalReward <- totalReward + double(reward).unwrapped()
            if isDone = true then
                break

            testState = State(newState).unwrapped()

        return totalReward



let iterationIndex = 0
let bestReward: double = 0.0
let trainEnvironment = gym.make("FrozenLake-v0")
let agent = Agent(trainEnvironment)
let testEnvironment = gym.make("FrozenLake-v0")
while true do
    if iterationIndex % 100 = 0 then
        print($"Running iteration {iterationIndex}")

    iterationIndex <- iterationIndex + 1
    let (state, action, reward, nextState) = agent.sampleEnvironment(trainEnvironment)
    agent.updateActionValue(state: state, action: action, reward=reward, nextState: nextState)

    let testReward: double = 0.0
    for _ in 0..<testEpisodeCount {
        testReward <- testReward + agent.playEpisode(testEnvironment: testEnvironment)

    testReward /= double(testEpisodeCount)
    if testReward > bestReward then
        print($"Best reward updated {bestReward} = {testReward}")
        bestReward = testReward

    if testReward > 0.80 then
        print($"Solved in {iterationIndex} iterations!")
        break


