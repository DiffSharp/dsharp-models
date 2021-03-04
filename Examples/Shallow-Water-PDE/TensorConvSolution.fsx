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

#r @"..\..\bin\Debug\net5.0\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\net5.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\net5.0\publish\Library.dll"

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
/// Discretization of the operator is implemented as convolution with a fixed 3x3 kernel. The
/// weights of the kernel are set in such a way so it matches the five-point stencil finite-difference
/// schema.
///
/// - Bug:
///  Result of applying Laplacian to the water height at a particular time-step is needlessly
///  (due to my laziness ;) calculated twice. When the time advances `u1` becomes `u0` of
///  the following time step. The result of applying the Laplace operator to `u1` can be cached
///  and reused a step later.
///
type TensorConvSolution(u0: Tensor, u1: Tensor, t: double) =
    
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

    /// Height of the water surface at previous time-step `t - Δt`
    do assert(u0.ndims = 2)
    do assert(u0.shape.[0] = resolution && u0.shape.[1] = resolution)

    /// Convolution kernel of the Laplace operator
    let laplaceKernel = dsharp.tensor([0.0, 1.0, 0.0, 
                                       1.0, -4.0, 1.0, 
                                       0.0, 1.0, 0.0], device=u0.device).view([3; 3; 1; 1])

    /// Applies discretized Laplace operator to scalar field `u`.
    let Δ(u: Tensor) : Tensor =
        let finiteDifference = dsharp.depthwiseConv2d(u.unsqueeze([0; 3]), filters=laplaceKernel, stride=1 (* , padding="valid" *))
        let Δu = finiteDifference / Δx / Δx
        Δu.squeeze([0; 3])

    /// Calculates a new solution stepped forward by one time-step `Δt`.
    ///
    /// - `u0` - Water surface height at previous time step
    /// - `u1` - Water surface height at current time step
    /// - `u2` - Water surface height at next time step (calculated)
    member _.Evolved() = 
        let Δu0 = Δ(u0).pad([1; 1])
        let Δu1 = Δ(u1).pad([1; 1])

        let Δu0Coefficient = c * α * Δt
        let Δu1Coefficient = c * c * Δt * Δt + c * α * Δt
        let cΔu0 = Δu0Coefficient * Δu0
        let cΔu1 = Δu1Coefficient * Δu1

        let u1twice = 2.0 * u1
        let u2 = u1twice + cΔu1 - u0 - cΔu0

        //LazyTensorBarrier(wait: true)
        TensorConvSolution(u0=u1, u1=u2, t=t + Δt)

    /// Water level height
    member _.WaterLevel = u1.unstack() |> Array.map (fun t -> t.toArray())

    /// Solution time
    member _.Time = t

    /// Calculates mean squared error loss between the solution and a `target` grayscale image.
    member _.meanSquaredError(target: Tensor) =
        assert(target.ndims = 2)
        assert(target.shape.[0] = resolution && target.shape.[1] = resolution)

        let error = u1 - target
        error.sqr().mean().toFloat32()

    /// Creates initial solution with water level `u0` at time `t`.
    new (waterLevel: Tensor) = TensorConvSolution(u0=waterLevel, u1=waterLevel, t=0.0)