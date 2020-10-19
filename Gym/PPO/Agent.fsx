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

/// Agent that uses the Proximal Policy Optimization (PPO).
///
/// Proximal Policy Optimization is an algorithm that trains an actor (policy) and a critic (value
/// function) using a clipped objective function. The clipped objective function simplifies the
/// update equation from its predecessor Trust Region Policy Optimization (TRPO). For more
/// information, check Proximal Policy Optimization Algorithms (Schulman et al., 2017).
type PPOAgent {
    // Cache for trajectory segments for minibatch updates.
    let memory: PPOMemory
    /// The learning rate for both the actor and the critic.
    let learningRate: double
    /// The discount factor that measures how much to weight to give to future
    /// rewards when calculating the action value.
    let discount: double
    /// Number of epochs to run minibatch updates once enough trajectory segments are collected.
    let epochs: int
    /// Parameter to clip the probability ratio.
    let clipEpsilon: double
    /// Coefficient for the entropy bonus added to the objective.
    let entropyCoefficient: double

    let actorCritic: ActorCritic
    let oldActorCritic: ActorCritic
    let actorOptimizer: Adam<ActorNetwork>
    let criticOptimizer: Adam<CriticNetwork>

    init(
        observationSize: int,
        hiddenSize: int,
        actionCount: int,
        learningRate: double,
        discount: double,
        epochs: int,
        clipEpsilon: double,
        entropyCoefficient: double
    ) = 
        self.learningRate = learningRate
        self.discount = discount
        self.epochs = epochs
        self.clipEpsilon = clipEpsilon
        self.entropyCoefficient = entropyCoefficient

        self.memory = PPOMemory()

        self.actorCritic = ActorCritic(
            observationSize: observationSize,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        self.oldActorCritic = self.actorCritic
        self.actorOptimizer = Adam(actorCritic.actorNetwork, learningRate: learningRate)
        self.criticOptimizer = Adam(actorCritic.criticNetwork, learningRate: learningRate)


    let step(env: PythonObject, state: PythonObject) = (PythonObject, Bool, Float) = 
        let tfState: Tensor = Tensor<Float>(numpy: np.array([state], dtype: np.float32))!
        let dist: Categorical<int32> = oldActorCritic(tfState)
        let action: int32 = dist.sample().scalarized()
        let (newState, reward, isDone, _) = env.step(action).tuple4

        memory.append(
            state: Array(state)!,
            action: action,
            reward: double(reward)!,
            logProb: dist.logProbabilities[int(action)].scalarized(),
            isDone: bool(isDone)!
        )

        return (newState, Bool(isDone)!, double(reward)!)


    let update() = 
        // Discount rewards for advantage estimation
        let rewards: double[] = []
        let discountedReward: double = 0
        for i in (0..<memory.rewards.count).reversed() = 
            if memory.isDones[i] then
                discountedReward = 0

            discountedReward = memory.rewards[i] + (discount * discountedReward)
            rewards.insert(discountedReward, at: 0)

        let tfRewards = Tensor<Float>(rewards)
        tfRewards = (tfRewards - tfRewards.mean()) / (tfRewards.stddev() + 1e-5)

        // Retrieve stored states, actions, and log probabilities
        let oldStates: Tensor = Tensor<Float>(numpy: np.array(memory.states, dtype: np.float32))!
        let oldActions: Tensor (*<int32>*) = Tensor (*<int32>*)(numpy: np.array(memory.actions, dtype: np.int32))!
        let oldLogProbs: Tensor = Tensor<Float>(numpy: np.array(memory.logProbs, dtype: np.float32))!

        // Optimize actor and critic
        let actorLosses: double[] = []
        let criticLosses: double[] = []
        for _ in 0..<epochs {
            // Optimize policy network (actor)
            let (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNetwork) =  actorNetwork -> Tensor<Float> in
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
                let entropyBonus: Tensor = Tensor<Float>(self.entropyCoefficient * dist.entropy())
                let loss: Tensor = -1 * (surrogateObjective + entropyBonus)

                return loss.mean()

            self.actorOptimizer.update(&self.actorCritic.actorNetwork, along: actorGradients)
            actorLosses.append(actorLoss.scalarized())

            // Optimize value network (critic)
            let (criticLoss, criticGradients) = valueWithGradient(at: self.actorCritic.criticNetwork) =  criticNetwork -> Tensor<Float> in
                let stateValues = criticNetwork(oldStates).flattened()
                let loss: Tensor = 0.5 * pow(stateValues - tfRewards, 2)

                return loss.mean()

            self.criticOptimizer.update(&self.actorCritic.criticNetwork, along: criticGradients)
            criticLosses.append(criticLoss.scalarized())

        self.oldActorCritic = self.actorCritic
        memory.removeAll()


