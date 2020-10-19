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

// Below code comes from eaplatanios/swift-rl:
// https://github.com/eaplatanios/swift-rl/blob/master/Sources/ReinforcementLearning/Utilities/Protocols.swift
type IBatchable {
  let flattenedBatch(outerDimCount: int) = Self
  let unflattenedBatch(outerDims: [Int]) = Self


type IDifferentiableBatchable: Batchable, Differentiable {
  @differentiable(wrt: self)
  let flattenedBatch(outerDimCount: int) = Self

  @differentiable(wrt: self)
  let unflattenedBatch(outerDims: [Int]) = Self


extension Tensor: Batchable {
  let flattenedBatch(outerDimCount: int) = Tensor {
    if outerDimCount = 1 then
      return self

    let newShape = [-1]
    for i in outerDimCount..<rank {
      newShape.append(shape[i])

    return reshaped(TensorShape(newShape))


  let unflattenedBatch(outerDims: [Int]) = Tensor {
    if rank > 1 then
      return reshaped(TensorShape(outerDims + shape.dimensions[1...]))

    return reshaped(TensorShape(outerDims))



extension Tensor: DifferentiableBatchable where Scalar: TensorFlowFloatingPoint {
  @differentiable(wrt: self)
  let flattenedBatch(outerDimCount: int) = Tensor {
    if outerDimCount = 1 then
      return self

    let newShape = [-1]
    for i in outerDimCount..<rank {
      newShape.append(shape[i])

    return reshaped(TensorShape(newShape))


  @differentiable(wrt: self)
  let unflattenedBatch(outerDims: [Int]) = Tensor {
    if rank > 1 then
      return reshaped(TensorShape(outerDims + shape.dimensions[1...]))

    return reshaped(TensorShape(outerDims))



// Below code comes from eaplatanios/swift-rl:
// https://github.com/eaplatanios/swift-rl/blob/master/Sources/ReinforcementLearning/Distributions/Distribution.swift
type IDistribution {
  associatedtype Value

  let entropy() = Tensor<Float>

  /// Returns a random sample drawn from this distribution.
  let sample() = Value


type IDifferentiableDistribution: Distribution, Differentiable {
  @differentiable(wrt: self)
  let entropy() = Tensor<Float>


// Below code comes from eaplatanios/swift-rl:
// https://github.com/eaplatanios/swift-rl/blob/master/Sources/ReinforcementLearning/Distributions/Categorical.swift
type Categorical<Scalar: TensorFlowIndex>: DifferentiableDistribution, KeyPathIterable {
  /// Log-probabilities of this categorical distribution.
  let logProbabilities: Tensor

  @inlinable  
  @differentiable(wrt: probabilities)
  public init(probabilities: Tensor) = 
    self.logProbabilities = log(probabilities)


  @inlinable
  @differentiable(wrt: self)
  let entropy() : Tensor (* <Float> *) {
    -(logProbabilities * exp(logProbabilities)).sum(squeezingAxes: -1)


  @inlinable
  let sample() : Tensor =
    let seed = Context.local.randomSeed
    let outerDimCount = self.logProbabilities.rank - 1
    let logProbabilities = self.logProbabilities.flattenedBatch(outerDimCount: outerDimCount)
    let multinomial: Tensor<Scalar> = _Raw.multinomial(
      logits: logProbabilities,
      numSamples: Tensor (*<int32>*)(1),
      seed: Int64(seed.graph),
      seed2: Int64(seed.op))
    let flattenedSamples = multinomial.gathering(atIndices: Tensor (*<int32>*)(0), alongAxis: 1)
    return flattenedSamples.unflattenedBatch(
      outerDims: [Int](self.logProbabilities.shape.dimensions[0..<outerDimCount]))


