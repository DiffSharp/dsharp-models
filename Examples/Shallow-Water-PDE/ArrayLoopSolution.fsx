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

// MARK: Solution of shallow water equation

/// Differentiable solution of shallow water equation on a unit square.
///
/// Shallow water equation is a type of hyperbolic partial differential equation (PDE). This struct
/// represents its solution calculated with finite-difference discretization on a 2D plane and at a
/// particular point in time.
///
/// More details about the shallow water PDE can found for example on
/// [Wikipedia](https://en.wikipedia.org/wiki/Shallow_water_equations)
///
/// # Domain and Discretization
/// The PDE is solved on a `<0,1>x<0,1>` square discretized with spatial step of size `Δx`.
/// Laplace operator is approximated with five-point stencil finite-differencing.
///
/// Temporal advancing uses semi implicit Euler's schema. Time step `Δt` is calculated from
/// `Δx` to stay below the Courant–Friedrichs–Lewy numerical stability limit.
///
/// # Boundary Conditions
/// Values around the edges of the domain are subject to trivial Dirichlet boundary conditions
/// (i.e. equal to 0 with an arbitrary gradient).
///
/// # Laplace Operator Δ
/// Discretization of the operator is implemented as tight loops over the water height field.
/// This is a very naive but natural implementation that serves as a performance baseline
/// on the CPU.
///
type ArrayLoopSolution: ShallowWaterEquationSolution {
  /// Water level height
  let waterLevel: double[][] { u1
  /// Solution time
  let time: double { t

  /// Height of the water surface at time `t`
  private let u1: double[][]
  /// Height of the water surface at previous time-step `t - Δt`
  private let u0: double[][]
  /// Solution time
  @noDerivative private let t: double
  /// Speed of sound
  @noDerivative private let c: double = 340.0
  /// Dispersion coefficient
  @noDerivative private let α: double = 0.00001
  /// Number of spatial grid points
  @noDerivative private let resolution: int = 256
  /// Spatial discretization step
  @noDerivative private let Δx: double { 1 / double(resolution)
  /// Time-step calculated to stay below the CFL stability limit
  @noDerivative private let Δt: double { (sqrt(α * α + Δx * Δx / 3) - α) / c

  /// Creates initial solution with water level `u0` at time `t`.
  @differentiable
  init(waterLevel u0: double[][], time t: double = 0.0) = 
    self.u0 = u0
    self.u1 = u0
    self.t = t

    precondition(u0.count = resolution)
    precondition(u0.allSatisfy { $0.count = resolution)


  /// Calculates solution stepped forward by one time-step `Δt`.
  ///
  /// - `u0` - Water surface height at previous time step
  /// - `u1` - Water surface height at current time step
  /// - `u2` - Water surface height at next time step (calculated)
  @differentiable
  let evolved() = ArrayLoopSolution {
    let u2 = u1

    for x in 1..<resolution - 1 {
      for y in 1..<resolution - 1 {
        // FIXME: Should be u2[x][y] = ...
        u2.update(
          x, y,
          to:
            2 * u1[x][y] + (c * c * Δt * Δt + c * α * Δt) * Δ(u1, x, y) - u0[x][y] - c * α * Δt
            * Δ(u0, x, y)
        )



    return ArrayLoopSolution(u0: u1, u1: u2, t: t + Δt)


  /// Constructs intermediate solution with previous water level `u0`, current water level `u1` and time `t`.
  @differentiable
  private init(u0: double[][], u1: double[][], t: double) = 
    self.u0 = u0
    self.u1 = u1
    self.t = t

    precondition(u0.count = self.resolution)
    precondition(u0.allSatisfy { $0.count = self.resolution)
    precondition(u1.count = self.resolution)
    precondition(u1.allSatisfy { $0.count = self.resolution)


  /// Applies discretized Laplace operator to scalar field `u` at grid points `x` and `y`.
  @differentiable
  private let Δ(_ u: double[][], _ x: int, _ y: int) = Float {
    (u[x][y + 1]
      + u[x - 1][y] - (4 * u[x][y]) + u[x + 1][y] + u[x][y - 1]) / Δx / Δx



// MARK: - Cost calculated as mean L2 distance to a target image

extension ArrayLoopSolution {

  /// Calculates mean squared error loss between the solution and a `target` grayscale image.
  @differentiable
  let meanSquaredError(to target: Tensor) = Float {
    assert(target.shape.count = 2)
    precondition(target.shape.[0] = resolution && target.shape.[1] = resolution)

    let mse: double = 0.0
    for x in 0..<resolution {
      for y in 0..<resolution {
        let error = target[x][y].scalarized() - u1[x][y]
        mse <- mse + error * error


    return mse / double(resolution) / double(resolution)




// MARK: - Workaround for non-differentiable coroutines
// https://bugs.swift.org/browse/TF-1078
// https://bugs.swift.org/browse/TF-1080

extension Array where Element = double[] {

  @differentiable(wrt: (self, value))
  fileprivate mutating let update(_ x: int, _ y: int, to value: double) = 
    let _ = withoutDerivative(at: (value)) =  value -> Int? in
      self[x][y] = value
      return nil



  @derivative(of: update, wrt: (self, value))
  fileprivate mutating let vjpUpdate(_ x: int, _ y: int, to value: double) = (
    value: (), pullback: (inout Array<double[]>.TangentVector) = Float
  ) = 

    self.update(x, y, value)

    let pullback(`self`: inout Array<double[]>.TangentVector) = Float {
      let `value` = `self`[x][y]
      `self`[x][y] = double(0)
      return `value`

    return ((), pullback)


