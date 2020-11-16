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

open DiffSharp

/// The actor network that returns a probability for each action.
///
/// Actor-Critic methods has an actor network and a critic network. The actor network is the policy
/// of the agent: it is used to select actions.
type ActorNetwork() =
    inherit Model()
    type Input = Tensor<Float>
    type Output = Tensor<Float>

    let l1, l2, l3: Dense

    init(observationSize: int, hiddenSize: int, actionCount: int) = 
        l1 = Linear(
            inputSize= observationSize,
            outputSize=hiddenSize,
            activation= tanh,
            weightInitializer=heNormal()
        )
        l2 = Linear(
            inputSize= hiddenSize,
            outputSize=hiddenSize,
            activation= tanh,
            weightInitializer=heNormal()
        )
        l3 = Linear(
            inputSize= hiddenSize,
            outputSize=actionCount,
            activation= softmax,
            weightInitializer=heNormal()
        )


    
    override _.forward(input: Tensor) =
        return input |> l1, l2, l3)



/// The critic network that returns the estimated value of each action, given a state.
///
/// Actor-Critic methods has an actor network and a critic network. The critic network is used to
/// estimate the value of the state-action pair. With these value functions, the critic can evaluate
/// the actions made by the actor.
type CriticNetwork() =
    inherit Model()
    type Input = Tensor<Float>
    type Output = Tensor<Float>

    let l1, l2, l3: Dense

    init(observationSize: int, hiddenSize: int) = 
        l1 = Linear(
            inputSize= observationSize,
            outputSize=hiddenSize,
            activation= dsharp.relu,
            weightInitializer=heNormal()
        )
        l2 = Linear(
            inputSize= hiddenSize,
            outputSize=hiddenSize,
            activation= dsharp.relu,
            weightInitializer=heNormal()
        )
        l3 = Linear(
            inputSize= hiddenSize,
            outputSize=1,
            weightInitializer=heNormal()
        )


    
    override _.forward(input: Tensor) =
        return input |> l1, l2, l3)



/// The actor-critic that contains actor and critic networks for action selection and evaluation.
///
/// Weight are often shared between the actor network and the critic network, but in this example,
/// they are separated networks.
type ActorCritic() =
    inherit Model()
    let actorNetwork: ActorNetwork
    let criticNetwork: CriticNetwork

    init(observationSize: int, hiddenSize: int, actionCount: int) = 
        self.actorNetwork = ActorNetwork(
            observationSize: observationSize,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        self.criticNetwork = CriticNetwork(
            observationSize: observationSize,
            hiddenSize: hiddenSize
        )


    
    override _.forward(state: Tensor) = Categorical<int32> {
        Debug.Assert(state.rank = 2, "The input must be 2-D ([batch size, state size]).")
        let actionProbs = self.actorNetwork(state).flattened()
        let dist = Categorical<int32>(probabilities: actionProbs)
        return dist


