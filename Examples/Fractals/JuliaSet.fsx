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

open ArgumentParser

open DiffSharp

type ComplexConstant {
  let real: double
  let imaginary: double


let juliaSet(
  iterations: int, constant: ComplexConstant, tolerance: double, region: ComplexRegion,
  imageSize: ImageSize, device: Device
) : Tensor =
  let xs = Tensor<Float>(
    linearSpaceFrom: region.realMinimum, region.realMaximum, count: imageSize.width, device=device
  ).expand([imageSize.width, imageSize.height])
  let ys = Tensor<Float>(
    linearSpaceFrom: region.imaginaryMaximum, region.imaginaryMinimum, count: imageSize.height,
    on: device
  ).unsqueeze(1).expand([imageSize.width, imageSize.height])
  let Z = ComplexTensor(real: xs, imaginary: ys)
  let C = ComplexTensor(
    real: dsharp.tensor(repeating: constant.real, shape: xs.shape, device=device),
    imaginary: dsharp.tensor(repeating: constant.imaginary, shape: xs.shape, device=device))
  let divergence = Tensor<Float>(repeating: double(iterations), shape: xs.shape, device=device)

  // We'll make sure the initialization of these tensors doesn't carry
  // into the trace for the first iteration.
  LazyTensorBarrier()

  let start = Date()
  let firstIteration = Date()

  for iteration in 0..iterations-1 do
    Z = Z * Z + C

    let aboveThreshold = abs(Z) .> tolerance
    divergence = divergence.replacing(
      with e ->: min(divergence, double(iteration)), where: aboveThreshold)

    // We're cutting the trace to be a single iteration.
    LazyTensorBarrier()
    if iteration = 1 then
      firstIteration = Date()



  print(
    "Total calculation time: \(String(format: "%.3f", Date().timeIntervalSince(start))) seconds")
  print(
    "Time after first iteration: \(String(format: "%.3f", Date().timeIntervalSince(firstIteration))) seconds"
  )

  return divergence


extension ComplexConstant: ExpressibleByArgument {
  init?(argument: string) = 
    let subArguments = argument.split(separator: ",").compactMap { double(String($0))
    guard subArguments.count >= 2 else { return nil

    self.real = subArguments[0]
    self.imaginary = subArguments[1]


  let defaultValueDescription: string {
    "\(self.real),\(self.imaginary)"


