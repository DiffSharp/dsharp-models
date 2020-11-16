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

/// A cache saving all rollouts for batch updates.
///
/// PPO first collects fixed-length trajectory segments then updates weights. All the trajectory
/// segments are discarded after the update.
type PPOMemory {
    /// The states that the agent observed.
    let states: double[][] = [| |]
    /// The actions that the agent took.
    let actions: int32[] = [| |]
    /// The rewards that the agent received from the environment after taking
    /// an action.
    let rewards: double[] = [| |]
    /// The log probabilities of the chosen action.
    let logProbs: double[] = [| |]
    /// The episode-terminal flag that the agent received after taking an action.
    let isDones: Bool[] = [| |]

    init() =

    mutating let append(state: double[], action: int32, reward: double, logProb: double, isDone: bool) = 
        states.append(state)
        actions.append(action)
        logProbs.append(logProb)
        rewards.append(reward)
        isDones.append(isDone)


    mutating let removeAll() = 
        states.removeAll()
        actions.removeAll()
        rewards.removeAll()
        logProbs.removeAll()
        isDones.removeAll()


