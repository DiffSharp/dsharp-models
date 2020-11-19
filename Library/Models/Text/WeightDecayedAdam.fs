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

namespace Models.Text

open DiffSharp
open DiffSharp.Optim

/// Adam optimizer with weight decay.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
type WeightDecayedAdam(?learningRate: double, ?beta1: double, ?beta2: double, ?weightDecayRate: double, ?epsilon: double) = 
  inherit ParameterGroupOptimizer()
  let learningRate = defaultArg learningRate 0.01
  let beta1 = defaultArg beta1 0.9
  let beta2 = defaultArg beta2 0.999
  let weightDecayRate = defaultArg weightDecayRate 0.01
  let epsilon = defaultArg epsilon 1e-6
  let b = ParameterGroupOptimizerBuilder()
  let learningRate = b.makeParameter("learningRate", dsharp.scalar learningRate)
  let beta1 = b.makeParameter("beta1", dsharp.scalar beta1)
  let beta2 = b.makeParameter("beta2", dsharp.scalar beta2)
  let wd = b.makeParameter("weightDecay", dsharp.scalar weightDecayRate)

  let firstMoment = b.makeStateParameter "firstMoment"
  let secondMoment = b.makeStateParameter "secondMoment"

  do 
      b.appendCallback (fun (state: OptimizerWeightStepState, optState: OptimizerState) ->
        optState.[state, firstMoment] <-
          state.[beta1] * optState.[state, firstMoment] + state.grad * (1.0 - state.[beta1]))

      b.appendCallback (fun (state: OptimizerWeightStepState, optState: OptimizerState) ->
        optState.[state, secondMoment] <-
          state.[beta2] * optState.[state, secondMoment] + state.grad * state.grad * (1.0 - state.[beta2]))

      b.appendCallback (fun (state: OptimizerWeightStepState, optState: OptimizerState) ->
        let denominator = sqrt(optState.[state, secondMoment]) + epsilon
        let update = optState.[state, firstMoment] / denominator + state.weight * state.[wd]
        state.step <- -state.[learningRate] * update)

      b.makeOptimizer()

