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
type ArrayLoopSolution(u0: double[,], u1: double[,], t: double) =
    /// Speed of sound
    let c: double = 340.0
    /// Dispersion coefficient
    let α: double = 0.00001
    /// Number of spatial grid points
    let resolution: int = 256
    /// Spatial discretization step
    let Δx = 1.0 / double(resolution)
    /// Time-step calculated to stay below the CFL stability limit
    let Δt = (sqrt(α * α + Δx * Δx / 3.0) - α) / c

    /// Applies discretized Laplace operator to scalar field `u` at grid points `x` and `y`.
  
    let Δ(u: double[,], x: int, y: int) =
        (u.[x,y + 1]
          + u.[x - 1,y] - (4.0 * u.[x,y]) + u.[x + 1,y] + u.[x,y - 1]) / Δx / Δx

    /// Calculates solution stepped forward by one time-step `Δt`.
    ///
    /// - `u0` - Water surface height at previous time step
    /// - `u1` - Water surface height at current time step
    /// - `u2` - Water surface height at next time step (calculated)
    member _.Evolved() = 
        let u2 = Array2D.copy u1  // TODO avoid this copy

        for x in 1..resolution - 1 do
            for y in 1..resolution - 1 do
                u2.[x,y] <- 
                    2.0 * u1.[x,y] + (c * c * Δt * Δt + c * α * Δt) * Δ(u1, x, y) - u0.[x,y] - c * α * Δt
                    * Δ(u0, x, y)

        ArrayLoopSolution(u0=u1, u1=u2, t=t + Δt)

    /// Water level height
    member _.WaterLevel = u1

    /// Solution time
    member _.Time = t

    /// Creates initial solution with water level `u0` at time `t`.
    new (waterLevel: double[,]) = ArrayLoopSolution(u0=waterLevel, u1=waterLevel, t=0.0)
