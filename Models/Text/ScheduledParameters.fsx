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

open DiffSharp

/// Scheduled parameter that takes the current training step as input and returns the parameter
/// value to be used for training. This can be used for scheduling the learning rate parameter,
/// for example.
type IScheduledParameter {
    associatedtype Scalar: FloatingPoint

    /// Returns the parameter value for the specified training step.
    ///
    /// - Parameter step: Training step.
    override _.forward(forStep step: UInt64) = Scalar


/// Dummy parameter schedule that represents no schedule being used. This is useful as a
/// default value whenever a parameter schedule argument is used.
type FixedParameter<Scalar: FloatingPoint>: ScheduledParameter {
    let value: Scalar

    @inlinable
    public init(value: Scalar) = 
        self.value = value


    @inlinable
    override _.forward(forStep step: UInt64) = Scalar {
        value



extension FixedParameter: ExpressibleByFloatLiteral
where Scalar: _ExpressibleByBuiltinFloatLiteral {
    type FloatLiteralType = Scalar

    public init(floatLiteral value: Scalar) = 
        self.init(value)



/// Linearly decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter(forStep: step)
/// let decayed = initial + step * slope
/// let decayedParameter = max(lowerBound * initial, decayed)
/// ```
type LinearlyDecayedParameter<BaseParameter: ScheduledParameter>: ScheduledParameter {
    type Scalar = BaseParameter.Scalar

    let baseParameter: BaseParameter
    let slope: Scalar
    let lowerBound: Scalar
    let startStep: UInt64

    /// Creates a new linearly decayed parameter.
    ///
    /// - Parameters:
    ///   - baseParameter: Parameter to decay.
    ///   - slope: Slope of the linear decay.
    ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
    ///   - startStep: Step after which to start decaying the parameter.
    @inlinable
    public init(
        baseParameter: BaseParameter,
        slope: Scalar,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) = 
        self.baseParameter = baseParameter
        self.slope = slope
        self.lowerBound = lowerBound
        self.startStep = startStep


    @inlinable
    override _.forward(forStep step: UInt64) = Scalar {
        let parameter = baseParameter(forStep: step)
        if step < startStep then return parameter
        let step = step - startStep
        let decayed = parameter + Scalar(step) * slope
        return max(lowerBound * parameter, decayed)



/// Exponentially decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter(forStep: step)
/// let decay = decayRate ^ (step / decayStepCount)
/// let decayedParameter = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
/// where if `staircase = true`, then `step / decayStepCount` uses integer division and the decayed
/// parameter value follows a staircase function.
type ExponentiallyDecayedParameter<BaseParameter: ScheduledParameter>: ScheduledParameter
where BaseParameter.Scalar: ElementaryFunctions {
    type Scalar = BaseParameter.Scalar

    let baseParameter: BaseParameter
    let decayRate: Scalar
    let decayStepCount: UInt64
    let staircase: bool
    let lowerBound: Scalar
    let startStep: UInt64

    /// Creates a new exponentially decayed parameter.
    ///
    /// - Parameters:
    ///   - baseParameter: Parameter to decay.
    ///   - decayRate: Decay rate.
    ///   - decayStepCount: Decay step count.
    ///   - staircase: If `true`, the decay will occur at discrete intervals.
    ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
    ///   - startStep: Step after which to start decaying the parameter.
    @inlinable
    public init(
        baseParameter: BaseParameter,
        decayRate: Scalar,
        decayStepCount: UInt64,
        staircase: bool = false,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) = 
        self.baseParameter = baseParameter
        self.decayRate = decayRate
        self.decayStepCount = decayStepCount
        self.staircase = staircase
        self.lowerBound = lowerBound
        self.startStep = startStep


    @inlinable
    override _.forward(forStep step: UInt64) = Scalar {
        let parameter = baseParameter(forStep: step)
        if step < startStep then return parameter
        let step = step - startStep
        let power = Scalar(step) / Scalar(decayStepCount)
        let decay = Scalar.pow(decayRate, staircase ? power.rounded(.down) : power)
        return parameter * ((1 - lowerBound) * decay + lowerBound)



/// Reciprocal square root decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter(forStep: step)
/// let decay = decayFactor / sqrt(max(step, decayThreshold))
/// let decayedParameter = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
type RSqrtDecayedParameter<BaseParameter: ScheduledParameter>: ScheduledParameter
where BaseParameter.Scalar: ElementaryFunctions {
    type Scalar = BaseParameter.Scalar

    let baseParameter: BaseParameter
    let decayFactor: Scalar
    let decayThreshold: Scalar
    let lowerBound: Scalar
    let startStep: UInt64

    /// Creates a new reciprocal square root decayed parameter.
    ///
    /// - Parameters:
    ///   - baseParameter: Parameter to decay.
    ///   - decayFactor: Decay factor.
    ///   - decayThreshold: Decay threshold.
    ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
    ///   - startStep: Step after which to start decaying the parameter.
    @inlinable
    public init(
        baseParameter: BaseParameter,
        decayFactor: Scalar,
        decayThreshold: Scalar,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) = 
        self.baseParameter = baseParameter
        self.decayFactor = decayFactor
        self.decayThreshold = decayThreshold
        self.lowerBound = lowerBound
        self.startStep = startStep


    @inlinable
    override _.forward(forStep step: UInt64) = Scalar {
        let parameter = baseParameter(forStep: step)
        if step < startStep then return parameter
        let step = step - startStep
        let decay = decayFactor / Scalar.sqrt(max(Scalar(step), decayThreshold))
        return parameter * ((1 - lowerBound) * decay + lowerBound)



/// Cosine decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter(forStep: step)
/// let decay = 0.5 * (1 + cos(pi * min(step, cycleStepCount) / cycleStepCount))
/// let decayedParameter = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
type CosineDecayedParameter<BaseParameter: ScheduledParameter>: ScheduledParameter
where BaseParameter.Scalar: ElementaryFunctions {
    type Scalar = BaseParameter.Scalar

    let baseParameter: BaseParameter
    let cycleStepCount: UInt64
    let lowerBound: Scalar
    let startStep: UInt64

    /// Creates a new cosine decayed parameter.
    ///
    /// - Parameters:
    ///   - baseParameter: Parameter to decay.
    ///   - cycleStepCount: Cosine decay cycle in terms of number of steps.
    ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
    ///   - startStep: Step after which to start decaying the parameter.
    @inlinable
    public init(
        baseParameter: BaseParameter,
        cycleStepCount: UInt64,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) = 
        self.baseParameter = baseParameter
        self.cycleStepCount = cycleStepCount
        self.lowerBound = lowerBound
        self.startStep = startStep


    @inlinable
    override _.forward(forStep step: UInt64) = Scalar {
        let parameter = baseParameter(forStep: step)
        if step < startStep then return parameter
        let step = step - startStep
        let cosine = Scalar.cos(Scalar(min(step, cycleStepCount)))
        let decay = (1 + cosine) * Scalar.pi / Scalar(2 * cycleStepCount)
        return parameter * ((1 - lowerBound) * decay + lowerBound)



/// Cycle-linear 10x decayed parameter.
///
/// The decayed parameter is computed as follows:
/// ```
/// let initial = baseParameter(forStep: step)
/// let cyclePosition = 1 - abs((step % (2 * cycleStepCount) - cycleStepCount) / cycleStepCount)
/// let decay = (0.1 + cyclePosition) * 3
/// let decayedParameter = initial * ((1 - lowerBound) * decay + lowerBound)
/// ```
type CycleLinear10xDecayedParameter<
    BaseParameter: ScheduledParameter
>: ScheduledParameter {
    type Scalar = BaseParameter.Scalar

    let baseParameter: BaseParameter
    let cycleStepCount: UInt64
    let lowerBound: Scalar
    let startStep: UInt64

    /// Creates a new cycle-linear 10x decayed parameter.
    ///
    /// - Parameters:
    ///   - baseParameter: Learning rate to decay.
    ///   - cycleStepCount: Cycle-linear 10x decay cycle in terms of number of steps.
    ///   - lowerBound: Minimum decayed parameter value as a fraction of its base value.
    ///   - startStep: Step after which to start decaying the parameter.
    @inlinable
    public init(
        baseParameter: BaseParameter,
        cycleStepCount: UInt64,
        lowerBound: Scalar = Scalar(0),
        startStep: UInt64 = 0
    ) = 
        self.baseParameter = baseParameter
        self.cycleStepCount = cycleStepCount
        self.lowerBound = lowerBound
        self.startStep = startStep


    @inlinable
    override _.forward(forStep step: UInt64) = Scalar {
        let parameter = baseParameter(forStep: step)
        if step < startStep then return parameter
        let step = step - startStep
        let ratio = Scalar((step % (2 * cycleStepCount) - cycleStepCount)) / Scalar(cycleStepCount)
        let cyclePosition = 1 - abs(ratio)
        let decay = (1 / Scalar(10) + cyclePosition) * 3 // 10x difference in each cycle (0.3 - 3).
        return parameter * ((1 - lowerBound) * decay + lowerBound)



/// Linearly warmed-up parameter.
///
/// For the first `warmUpStepCount` steps the base parameter is multiplied with:
/// ```
/// warmUpOffset + ((1 - warmUpOffset) / warmUpStepCount) * step
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
type LinearlyWarmedUpParameter<BaseParameter: ScheduledParameter>: ScheduledParameter {
    type Scalar = BaseParameter.Scalar

    let baseParameter: BaseParameter
    let warmUpStepCount: UInt64
    let warmUpOffset: Scalar

    /// Creates a new linear parameter warm-up schedule.
    ///
    /// - Parameters:
    ///   - baseParameter: Parameter to warm-up.
    ///   - warmUpStepCount: Number of warm-up steps.
    ///   - warmUpOffset: Linear schedule offset.
    @inlinable
    public init(
        baseParameter: BaseParameter,
        warmUpStepCount: UInt64,
        warmUpOffset: Scalar
    ) = 
        self.baseParameter = baseParameter
        self.warmUpStepCount = warmUpStepCount
        self.warmUpOffset = warmUpOffset


    @inlinable
    override _.forward(forStep step: UInt64) = Scalar {
        let parameter = baseParameter(forStep: step)
        if step >= warmUpStepCount then return parameter
        let factor = warmUpOffset + ((1 - warmUpOffset) / Scalar(warmUpStepCount)) * Scalar(step)
        return parameter * factor



/// Exponentially warmed-up parameter.
///
/// For the first `warmUpStepCount` steps the base parameter is multiplied with:
/// ```
/// exp(log(warmUpFactor) / step) ^ (warmUpStepCount - step)
/// ```
///
/// - Source: [Attention is All You Need (Section 5.3)](https://arxiv.org/pdf/1706.03762.pdf).
type ExponentiallyWarmedUpParameter<BaseParameter: ScheduledParameter>: ScheduledParameter
where BaseParameter.Scalar: ElementaryFunctions {
    type Scalar = BaseParameter.Scalar

    let baseParameter: BaseParameter
    let warmUpStepCount: UInt64
    let warmUpFactor: Scalar

    /// Creates a new exponential parameter warm-up schedule.
    ///
    /// - Parameters:
    ///   - baseParameter: Parameter to warm-up.
    ///   - warmUpStepCount: Number of warm-up steps.
    ///   - warmUpFactor: Warm-up parameter scaling factor.
    @inlinable
    public init(
        baseParameter: BaseParameter,
        warmUpStepCount: UInt64,
        warmUpFactor: Scalar
    ) = 
        self.baseParameter = baseParameter
        self.warmUpStepCount = warmUpStepCount
        self.warmUpFactor = warmUpFactor


    @inlinable
    override _.forward(forStep step: UInt64) = Scalar {
        let parameter = baseParameter(forStep: step)
        if step >= warmUpStepCount then return parameter
        let base = Scalar.exp(Scalar.log(warmUpFactor) / Scalar(warmUpStepCount))
        let factor = Scalar.pow(base, Scalar(warmUpStepCount - step))
        return parameter * factor


