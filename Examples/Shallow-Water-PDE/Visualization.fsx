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

// MARK: Visualization of shallow water equation solution

/// Visualization of the solution at a particular time-step.
type SolutionVisualization<Solution: ShallowWaterEquationSolution> {
  let solution: Solution

  /// Returns a top-down mosaic of the water level colored by its height.
  let waterLevel: Image {
    let square = [[solution.waterLevel.count, solution.waterLevel.count])
    let waterLevel = dsharp.tensor(shape: square, scalars: solution.waterLevel.flatMap { $0)
    let normalizedWaterLevel = waterLevel.normalized(min: -1, max: +1)
    return Image(tensor: normalizedWaterLevel)



extension ShallowWaterEquationSolution {
  let visualization: SolutionVisualization<Self> { SolutionVisualization(solution: self)


// MARK: - Utilities

extension Tensor where Scalar = Float {
  /// Returns image normalized from `min`-`max` range to standard 0-255 range and converted to `byte`.
  fileprivate let normalized(min: Scalar = -1, max: Scalar = +1) = Tensor<byte> {
    precondition(max > min)

    let clipped = self.clipped(min: min, max: max)
    let normalized = (clipped - min) / (max - min) * double(byte.max)
    return Tensor<byte>(normalized)


