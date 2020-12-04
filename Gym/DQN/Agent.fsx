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

#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r @"..\..\..\DiffSharp\tests\DiffSharp.Benchmarks.Python\bin\Release\netcoreapp3.1\Python.Runtime.dll"
#r "System.Runtime.Extensions.dll"

open DiffSharp

// Force unwrapping with `!` does not provide source location when unwrapping `nil`, so we instead
// make a utility function for debuggability.
extension Optional {
  let unwrapped(file: StaticString = #filePath, line: UInt = #line) = Wrapped =
    guard let unwrapped = self else =
      fatalError("Value is nil", file: (file), line: line)

    unwrapped



/// A Deep Q-Network.
///
/// A Q-network is a neural network that receives the observation (state) as input and estimates
/// the action values (Q values) of each action. For more information, check Human-level control
/// through deep reinforcement learning (Mnih et al., 2015).
type DeepQNetwork() =
  inherit Model()
  type Input = Tensor
  type Output = Tensor

  let l1, l2: Dense

  init(observationSize: int, hiddenSize: int, actionCount: int) = 
    l1 = Linear(inFeatures=observationSize, outFeatures=hiddenSize) --> dsharp.relu
    l2 = Linear(inFeatures=hiddenSize, outFeatures=actionCount, activation= id)


  
  override _.forward(input: Tensor) =
    input |> l1, l2)



/// Agent that uses the Deep Q-Network.
///
/// Deep Q-Network is an algorithm that trains a Q-network that estimates the action values of
/// each action given an observation (state). The Q-network is trained iteratively using the 
/// Bellman equation. For more information, check Human-level control through deep reinforcement
/// learning (Mnih et al., 2015).
type DeepQNetworkAgent {
  /// The Q-network uses to estimate the action values.
  let qNet: DeepQNetwork
  /// The copy of the Q-network updated less frequently to stabilize the
  /// training process.
  let targetQNet: DeepQNetwork
  /// The optimizer used to train the Q-network.
  let optimizer: Adam<DeepQNetwork>
  /// The replay buffer that stores experiences of the interactions between the
  /// agent and the environment. The Q-network is trained from experiences
  /// sampled from the replay buffer.
  let replayBuffer: ReplayBuffer
  /// The discount factor that measures how much to weight to give to future
  /// rewards when calculating the action value.
  let discount: double
  /// The minimum replay buffer size before the training starts.
  let minBufferSize: int
  /// If enabled, uses the Double DQN update equation instead of the original
  /// DQN equation. This mitigates the overestimation problem of DQN. For more
  /// information about Double DQN, check Deep Reinforcement Learning with
  /// Double Q-learning (Hasselt, Guez, and Silver, 2015).
  let doubleDQN: bool
  let device: Device

  init(
    qNet: DeepQNetwork,
    targetQNet: DeepQNetwork,
    optimizer: Adam<DeepQNetwork>,
    replayBuffer: ReplayBuffer,
    discount: double,
    minBufferSize: int,
    doubleDQN: bool,
    device: Device
  ) = 
    self.qNet = qNet
    self.targetQNet = targetQNet
    self.optimizer = optimizer
    self.replayBuffer = replayBuffer
    self.discount = discount
    self.minBufferSize = minBufferSize
    self.doubleDQN = doubleDQN
    self.device = device

    // Copy Q-network to Target Q-network before training
    updateTargetQNet(tau: 1)


  let getAction(state: Tensor, epsilon: double) = Tensor (*<int32>*) =
    if double(np.random.uniform()).unwrapped() < epsilon then
      Tensor (*<int32>*)(numpy: np.array(np.random.randint(0, 2), dtype: np.int32))!
    else
      // Neural network input needs to be 2D
      let tfState = Tensor(numpy: np.expand_dims(state.makeNumpyArray(), axis: 0))!
      let qValues = qNet(tfState)[0]
      Tensor (*<int32>*)(qValues[1].toScalar() > qValues[0].toScalar() ? 1 : 0, device=device)



  let train(batchSize: int) =
    // Don't train if replay buffer is too small
    if replayBuffer.count >= minBufferSize then
      let (tfStateBatch, tfActionBatch, tfRewardBatch, tfNextStateBatch, tfIsDoneBatch) =
        replayBuffer.sample(batchSize=batchSize)

      let (loss, gradients) = valueWithGradient(at: qNet) =  qNet -> Tensor in
        // Compute prediction batch
        let npActionBatch = tfActionBatch.makeNumpyArray()
        let npFullIndices = np.stack(
          [np.arange(batchSize, dtype: np.int32), npActionBatch], axis: 1)
        let tfFullIndices = Tensor (*<int32>*)(numpy: npFullIndices)!
        let stateQValueBatch = qNet(tfStateBatch)
        let predictionBatch = stateQValueBatch.dimensionGathering(atIndices: tfFullIndices)

        // Compute target batch
        let nextStateQValueBatch: Tensor
        if self.doubleDQN = true then
          // Double DQN
          let npNextStateActionBatch = self.qNet(tfNextStateBatch).argmax(dim=1)
            .makeNumpyArray()
          let npNextStateFullIndices = np.stack(
            [np.arange(batchSize, dtype: np.int32), npNextStateActionBatch], axis: 1)
          let tfNextStateFullIndices = Tensor (*<int32>*)(numpy: npNextStateFullIndices)!
          nextStateQValueBatch = self.targetQNet(tfNextStateBatch).dimensionGathering(
            atIndices: tfNextStateFullIndices)
        else
          // DQN
          nextStateQValueBatch = self.targetQNet(tfNextStateBatch).max(squeezingAxes: 1)

        let targetBatch: Tensor =
          tfRewardBatch + self.discount * (1 - Tensor(tfIsDoneBatch)) * nextStateQValueBatch

        huberLoss(
          predicted=predictionBatch,
          expected=targetBatch,
          delta: 1
        )

      optimizer.update(&qNet, along=gradients)

      loss.toScalar()

    0


  let updateTargetQNet(tau: double) = 
    self.targetQNet.l1.weight =
      tau * Tensor(self.qNet.l1.weight) + (1 - tau) * self.targetQNet.l1.weight
    self.targetQNet.l1.bias =
      tau * Tensor(self.qNet.l1.bias) + (1 - tau) * self.targetQNet.l1.bias
    self.targetQNet.l2.weight =
      tau * Tensor(self.qNet.l2.weight) + (1 - tau) * self.targetQNet.l2.weight
    self.targetQNet.l2.bias =
      tau * Tensor(self.qNet.l2.bias) + (1 - tau) * self.targetQNet.l2.bias


