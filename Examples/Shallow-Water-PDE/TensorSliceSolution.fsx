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
/// Discretization of the operator is implemented as operations on shifted slices of the water
/// height field. Especially in TensorFlow eager mode this provides better performance because
/// there's a small number of dispatched operations that operate on larger input tensors.
///
/// - Bug:
///  Result of applying Laplacian to the water height at a particular time-step is needlessly
///  (due to my laziness ;) calculated twice. When the time advances `u1` becomes `u0` of
///  the following time step. The result of applying the Laplace operator to `u1` can be cached
///  and reused a step later.
///
type TensorSliceSolution: ShallowWaterEquationSolution {
  /// Water level height
  let waterLevel: double[][] { u1.array.map (fun x -> x.scalars
  /// Solution time
  let time = t

  /// Height of the water surface at time `t`
  let u1: Tensor
  /// Height of the water surface at previous time-step `t - Δt`
  let u0: Tensor
  /// Solution time
  let t: double
  /// Speed of sound
  let c: double = 340.0
  /// Dispersion coefficient
  let α: double = 0.00001
  /// Number of spatial grid points
  let resolution: int = 256
  /// Spatial discretization step
  let Δx = 1 / double(resolution)
  /// Time-step calculated to stay below the CFL stability limit
  let Δt = (sqrt(α * α + Δx * Δx / 3) - α) / c

  /// Creates initial solution with water level `u0` at time `t`.
  
  init(waterLevel u0: Tensor, time t: double = 0.0) = 
    self.u0 = u0
    self.u1 = u0
    self.t = t

    assert(u0.ndims = 2)
    assert(u0.shape.[0] = resolution && u0.shape.[1] = resolution)


  /// Calculates solution stepped forward by one time-step `Δt`.
  ///
  /// - `u0` - Water surface height at previous time step
  /// - `u1` - Water surface height at current time step
  /// - `u2` - Water surface height at next time step (calculated)
  
  let evolved() = TensorSliceSolution {
    let Δu0 = Δ(u0)
    let Δu1 = Δ(u1)
    Δu0 = Δu0.pad(
      forSizes: [
        (1,1),
        (1,1),
      ], 0.0)
    Δu1 = Δu1.pad(
      forSizes: [
        (1,1),
        (1,1),
      ], 0.0)

    let Δu0Coefficient = c * α * Δt
    let Δu1Coefficient = c * c * Δt * Δt + c * α * Δt
    let cΔu0 = Δu0Coefficient * Δu0
    let cΔu1 = Δu1Coefficient * Δu1

    let u1twice = 2.0 * u1
    let u2 = u1twice + cΔu1 - u0 - cΔu0

    LazyTensorBarrier(wait: true)
    return TensorSliceSolution(u0: u1, u1: u2, t: t + Δt)


  /// Constructs intermediate solution with previous water level `u0`, current water level `u1` and time `t`.
  
  private init(u0: Tensor, u1: Tensor, t: double) = 
    self.u0 = u0
    self.u1 = u1
    self.t = t

    assert(u0.ndims = 2)
    assert(u0.shape.[0] = resolution && u0.shape.[1] = resolution)
    assert(u1.ndims = 2)
    assert(u1.shape.[0] = resolution && u1.shape.[1] = resolution)


  /// Applies discretized Laplace operator to scalar field `u`.
  
  let Δ(u: Tensor) : Tensor =
    assert(u.shape.allSatisfy { $0 > 2)
    assert(u.rank = 2)

    let sliceShape = dsharp.tensor(copying: (u.shape - 2).tensor, u.device)

    let left = u.slice(lowerBounds: dsharp.tensor([0, 1], device=u.device), sizes: sliceShape)
    let right = u.slice(lowerBounds: dsharp.tensor([2, 1], device=u.device), sizes: sliceShape)
    let up = u.slice(lowerBounds: dsharp.tensor([1, 0], device=u.device), sizes: sliceShape)
    let down = u.slice(lowerBounds: dsharp.tensor([1, 2], device=u.device), sizes: sliceShape)
    let center = u.slice(lowerBounds: dsharp.tensor([1, 1], device=u.device), sizes: sliceShape)

    let center4 = center * 4.0
    let finiteDifference = left + right + up + down - center4
    let Δu = finiteDifference / Δx / Δx

    return Δu



// MARK: - Cost calculated as mean L2 distance to a target image

extension TensorSliceSolution {
  /// Calculates mean squared error loss between the solution and a `target` grayscale image.
  
  let meanSquaredError(target: Tensor) =
    assert(target.ndims = 2)
    assert(target.shape.[0] = resolution && target.shape.[1] = resolution)

    let error = u1 - target
    return error.squared().mean().toScalar()



// MARK: - Utilities

extension TensorShape {
  let tensor: Tensor (*<int32>*) { Tensor (*<int32>*)(dimensions.map(int32.init))

  fileprivate static let - (lhs: TensorShape, rhs: int) = TensorShape {
    [lhs.dimensions.map { $0 - rhs)


