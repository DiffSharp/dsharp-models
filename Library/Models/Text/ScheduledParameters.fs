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

namespace Models

open System
open DiffSharp

/// Scheduled parameter that takes the current training step as input and returns the parameter
/// value to be used for training. This can be used for scheduling the learning rate parameter,
/// for example.
[<AbstractClass>]
type ScheduledParameter() =

    /// Returns the parameter value for the specified training step.
    ///
    /// - Parameter step: Training step.
    abstract forward : step: uint64 -> double

/// Dummy parameter schedule that represents no schedule being used. This is useful as a
/// default value whenever a parameter schedule argument is used.
type FixedParameter(value: double) =
    inherit ScheduledParameter()

    override _.forward(step: uint64) = value

/// Linearly decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter.forward(step)
/// let decayed = initial + step * slope
/// let decayedParameter = max(lowerBound * initial, decayed)
/// ```
/// Creates a new linearly decayed parameter.
///
/// - Parameters:
///   - baseParameter: Parameter to decay.
///   - slope: Slope of the linear decay.
///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
///   - startStep: Step after which to start decaying the parameter.
type LinearlyDecayedParameter(baseParameter: ScheduledParameter,
        slope: double,
        ?lowerBound: double,
        ?startStep: uint64) =

    inherit ScheduledParameter()
    let lowerBound = defaultArg lowerBound 0.0
    let startStep = defaultArg startStep 0UL

    override _.forward(step: uint64) =
        let parameter = baseParameter.forward(step)
        if step < startStep then parameter else
        let step = step - startStep
        let decayed = parameter + float step * slope
        max (lowerBound.toDouble() * parameter.toDouble()) decayed

/// Exponentially decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter.forward(step)
/// let decay = decayRate ^ (step / decayStepCount)
/// let decayedParameter = initial * ((1.0 - lowerBound) * decay + lowerBound)
/// ```
/// where if `staircase = true`, then `step / decayStepCount` uses integer division and the decayed
/// parameter value follows a staircase function.
///
/// Creates a new exponentially decayed parameter.
///
/// - Parameters:
///   - baseParameter: Parameter to decay.
///   - decayRate: Decay rate.
///   - decayStepCount: Decay step count.
///   - staircase: If `true`, the decay will occur at discrete intervals.
///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
///   - startStep: Step after which to start decaying the parameter.
type ExponentiallyDecayedParameter(baseParameter: ScheduledParameter,
        decayRate: float,
        decayStepCount: uint64,
        staircase: bool,
        ?lowerBound: float,
        ?startStep: uint64) =
    inherit ScheduledParameter()
    let lowerBound = defaultArg lowerBound 0.0
    let startStep = defaultArg startStep 0UL

    override _.forward(step: uint64) =
        let parameter = baseParameter.forward(step)
        if step < startStep then parameter else
        let step = step - startStep
        let power = float step / float decayStepCount
        let decay = float decayRate ** (if staircase then floor power else power)
        parameter * ((1.0 - lowerBound) * decay + lowerBound)

/// Reciprocal square root decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter.forward(step)
/// let decay = decayFactor / sqrt(max(step, decayThreshold))
/// let decayedParameter = initial * ((1.0 - lowerBound) * decay + lowerBound)
/// ```
type RSqrtDecayedParameter(baseParameter: ScheduledParameter,
        decayFactor: float,
        decayThreshold: float,
        ?lowerBound: float,
        ?startStep: uint64) =

    inherit ScheduledParameter()
    let lowerBound = defaultArg lowerBound 0.0
    let startStep = defaultArg startStep 0UL

    /// Creates a new reciprocal square root decayed parameter.
    ///
    /// - Parameters:
    ///   - baseParameter: Parameter to decay.
    ///   - decayFactor: Decay factor.
    ///   - decayThreshold: Decay threshold.
    ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
    ///   - startStep: Step after which to start decaying the parameter.

    override _.forward(step: uint64) =
        let parameter = baseParameter.forward(step)
        if step < startStep then parameter else
        let step = step - startStep
        let decay = decayFactor / sqrt(max (double step) decayThreshold)
        parameter * ((1.0 - lowerBound) * decay + lowerBound)

/// Cosine decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter.forward(step)
/// let decay = 0.5 * (1.0 + cos(pi * (min step, cycleStepCount) / cycleStepCount))
/// let decayedParameter = initial * ((1.0 - lowerBound) * decay + lowerBound)
/// ```
/// Creates a new cosine decayed parameter.
///
/// - Parameters:
///   - baseParameter: Parameter to decay.
///   - cycleStepCount: Cosine decay cycle in terms of number of steps.
///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
///   - startStep: Step after which to start decaying the parameter.

type CosineDecayedParameter(baseParameter: ScheduledParameter,
        cycleStepCount: uint64,
        ?lowerBound: float,
        ?startStep: uint64
    ) =

    inherit ScheduledParameter()
    let lowerBound = defaultArg lowerBound 0.0
    let startStep = defaultArg startStep 0UL
    override _.forward(step: uint64) =
        let parameter = baseParameter.forward(step)
        if step < startStep then parameter else
        let step = step - startStep
        let cosine = cos(double (min step cycleStepCount))
        let decay = (1.0 + cosine) * Math.PI / double (2UL * cycleStepCount)
        parameter * ((1.0 - lowerBound) * decay + lowerBound)

/// Cycle-linear 10x decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter.forward(step)
/// let cyclePosition = 1 - abs((step % (2.- * cycleStepCount) - cycleStepCount) / cycleStepCount)
/// let decay = (0.1.0 + cyclePosition) * 3
/// let decayedParameter = initial * ((1.0 - lowerBound) * decay + lowerBound)
/// ```
///
/// Creates a new cycle-linear 10x decayed parameter.
///
/// - Parameters:
///   - baseParameter: Learning rate to decay.
///   - cycleStepCount: Cycle-linear 10x decay cycle in terms of number of steps.
///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
///   - startStep: Step after which to start decaying the parameter.
type CycleLinear10xDecayedParameter(baseParameter: ScheduledParameter,
        cycleStepCount: uint64,
        ?lowerBound: float,
        ?startStep: uint64) =

    inherit ScheduledParameter()
    let lowerBound = defaultArg lowerBound 0.0
    let startStep = defaultArg startStep 0UL
    override _.forward(step: uint64) =
        let parameter = baseParameter.forward(step)
        if step < startStep then parameter else
        let step = step - startStep
        let ratio = (double (step % (2UL * cycleStepCount) - cycleStepCount)) / (double cycleStepCount)
        let cyclePosition = 1.0 - abs(ratio)
        let decay = (1.0 / 10.0 + cyclePosition) * 3.0 // 10x difference in each cycle (0.3 - 3).
        parameter * ((1.0 - lowerBound) * decay + lowerBound)

/// Linearly warmed-up parameter.
///
/// For the first `warmUpStepCount` steps the base parameter is multiplied with:
/// ```
/// warmUpOffset + ((1 - warmUpOffset) / warmUpStepCount) * step
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
///
/// Creates a new linear parameter warm-up schedule.
///
/// - Parameters:
///   - baseParameter: Parameter to warm-up.
///   - warmUpStepCount: Number of warm-up steps.
///   - warmUpOffset: Linear schedule offset.

type LinearlyWarmedUpParameter(baseParameter: ScheduledParameter,
        warmUpStepCount: uint64,
        warmUpOffset: float) =
    
    inherit ScheduledParameter()
    override _.forward(step: uint64) =
        let parameter = baseParameter.forward(step)
        if step >= warmUpStepCount then parameter else
        let factor = warmUpOffset + ((1.0 - warmUpOffset) / (double warmUpStepCount)) * (double step)
        parameter * factor

/// Exponentially warmed-up parameter.
///
/// For the first `warmUpStepCount` steps the base parameter is multiplied with:
/// ```
/// exp(log(warmUpFactor) / step) ^ (warmUpStepCount - step)
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
///
/// Creates a new exponential parameter warm-up schedule.
///
/// - Parameters:
///   - baseParameter: Parameter to warm-up.
///   - warmUpStepCount: Number of warm-up steps.
///   - warmUpFactor: Warm-up parameter scaling factor.
type ExponentiallyWarmedUpParameter(baseParameter: ScheduledParameter,
        warmUpStepCount: uint64,
        warmUpFactor: float
    ) =

    inherit ScheduledParameter()
    override _.forward(step: uint64) =
        let parameter = baseParameter.forward(step)
        if step >= warmUpStepCount then parameter else
        let v = exp(log(warmUpFactor) / (double warmUpStepCount))
        let factor = v ** (double warmUpStepCount - double step)
        parameter * factor
