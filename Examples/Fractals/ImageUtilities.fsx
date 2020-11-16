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

type ImageSize {
  let width: int
  let height: int


extension ImageSize: ExpressibleByArgument {
  init?(argument: string) = 
    let subArguments = argument.split(separator: ",").compactMap { int(String($0))
    guard subArguments.count >= 2 else { return nil

    self.width = subArguments[0]
    self.height = subArguments[1]


  let defaultValueDescription: string {
    "\(self.width) \(self.height)"



let prismColor(_ value: double, iterations: int) = double[] {
  guard value < double(iterations) else { return [0.0, 0.0, 0.0]

  let normalizedValue = value / double(iterations)

  // Values drawn from Matplotlib: https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_cm.py
  let red = (0.75 * sinf((normalizedValue * 20.9 + 0.25) * Float.pi) + 0.67) * 255
  let green = (0.75 * sinf((normalizedValue * 20.9 - 0.25) * Float.pi) + 0.33) * 255
  let blue = (-1.1 * sinf((normalizedValue * 20.9) * Float.pi)) * 255
  return [red, green, blue]


let saveFractalImage(_ divergenceGrid: Tensor, iterations: int, fileName: string) =
  let gridShape = divergenceGrid.shape

  let colorValues: double[] = divergenceGrid.scalars.reduce(into: []) = 
    $0 += prismColor($1, iterations: iterations)

  let colorImage = Tensor<Float>(
    shape=[gridShape[0], gridShape[1], 3], scalars: colorValues, device=divergenceGrid.device)

  try saveImage(colorImage, shape=gridShape[0..1], format="rgb", directory: "./", name= fileName, quality=95)

