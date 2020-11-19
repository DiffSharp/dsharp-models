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

open PythonKit
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim

/// Agent that uses the Proximal Policy Optimization (PPO).
///
/// Proximal Policy Optimization is an algorithm that trains an actor (policy) and a critic (value
/// function) using a clipped objective function. The clipped objective function simplifies the
/// update equation from its predecessor Trust Region Policy Optimization (TRPO). For more
/// information, check Proximal Policy Optimization Algorithms (Schulman et al., 2017).
type PPOAgent(
        observationSize: int,
        hiddenSize: int,
        actionCount: int,
        learningRate: double,
        discount: double,
        epochs: int,
        clipEpsilon: double,
        entropyCoefficient: double
    ) =
    /// The learning rate for both the actor and the critic.
    let learningRate = learningRate
    /// The discount factor that measures how much to weight to give to future
    /// rewards when calculating the action value.
    let discount = discount
    /// Number of epochs to run minibatch updates once enough trajectory segments are collected.
    let epochs = epochs
    /// Parameter to clip the probability ratio.
    let clipEpsilon = clipEpsilon
    /// Coefficient for the entropy bonus added to the objective.
    let entropyCoefficient = entropyCoefficient

    // Cache for trajectory segments for minibatch updates.
    let memory = PPOMemory()

    let actorCritic = ActorCritic(
            observationSize=observationSize,
            hiddenSiz=hiddenSize,
            actionCount=actionCount
        )
    let mutable oldActorCritic = actorCritic
    let actorOptimizer = Adam(actorCritic.actorNetwork, learningRate: learningRate)
    let criticOptimizer = Adam(actorCritic.criticNetwork, learningRate: learningRate)
    let step(env: PythonObject, state: PythonObject) : (PythonObject * Bool * Float) =
        let tfState: Tensor = Tensor(numpy: np.array([state], dtype: np.float32))!
        let dist: Categorical<int32> = oldActorCritic(tfState)
        let action: int32 = dist.sample().toScalar()
        let (newState, reward, isDone, _) = env.step(action).tuple4

        memory.append(
            state: Array(state)!,
            action: action,
            reward: double(reward)!,
            logProb: dist.logProbabilities[int(action)].toScalar(),
            isDone: bool(isDone)!
        )

        (newState, Bool(isDone)!, double(reward)!)


    let update() = 
        // Discount rewards for advantage estimation
        let rewards: double[] = [| |]
        let discountedReward: double = 0
        for i in (0..<memory.rewards.count).reversed() do            if memory.isDones[i] then
                discountedReward = 0

            discountedReward = memory.rewards[i] + (discount * discountedReward)
            rewards.insert(discountedReward, at: 0)

        let tfRewards = Tensor(rewards)
        tfRewards = (tfRewards - tfRewards.mean()) / (tfRewards.stddev() + 1e-5)

        // Retrieve stored states, actions, and log probabilities
        let oldStates: Tensor = Tensor(numpy: np.array(memory.states, dtype: np.float32))!
        let oldActions: Tensor (*<int32>*) = Tensor (*<int32>*)(numpy: np.array(memory.actions, dtype: np.int32))!
        let oldLogProbs: Tensor = Tensor(numpy: np.array(memory.logProbs, dtype: np.float32))!

        // Optimize actor and critic
        let actorLosses: double[] = [| |]
        let criticLosses: double[] = [| |]
        for _ in 0..epochs-1 do
            // Optimize policy network (actor)
            let (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNetwork) =  actorNetwork -> Tensor in
                let npIndices = np.stack([np.arange(oldActions.shape.[0], dtype: np.int32), oldActions.makeNumpyArray()], axis: 1)
                let tfIndices = Tensor (*<int32>*)(numpy: npIndices)!
                let actionProbs = actorNetwork(oldStates).dimensionGathering(atIndices: tfIndices)

                let dist = Categorical<int32>(probabilities: actionProbs)
                let stateValues = self.actorCritic.criticNetwork(oldStates).flattened()
                let ratios: Tensor = exp(dist.logProbabilities - oldLogProbs)

                let advantages: Tensor = tfRewards - stateValues
                let surrogateObjective = dsharp.tensor(stacking: [
                    ratios * advantages,
                    ratios.clipped(min:1 - self.clipEpsilon, max: 1 + self.clipEpsilon) * advantages
                ]).min(dim=0).flattened()
                let entropyBonus: Tensor = Tensor(self.entropyCoefficient * dist.entropy())
                let loss: Tensor = -1 * (surrogateObjective + entropyBonus)

                loss.mean()

            self.actorOptimizer.update(&self.actorCritic.actorNetwork, along=actorGradients)
            actorLosses.append(actorLoss.toScalar())

            // Optimize value network (critic)
            let (criticLoss, criticGradients) = valueWithGradient(at: self.actorCritic.criticNetwork) =  criticNetwork -> Tensor in
                let stateValues = criticNetwork(oldStates).flattened()
                let loss: Tensor = 0.5 * pow(stateValues - tfRewards, 2)

                loss.mean()

            self.criticOptimizer.update(&self.actorCritic.criticNetwork, along=criticGradients)
            criticLosses.append(criticLoss.toScalar())

        self.oldActorCritic = self.actorCritic
        memory.removeAll()


