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

type FractalCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "Fractals",
    abstract: """
      Computes fractals of a variety of types and writes an image from the results.
      """,
    subcommands: [
      JuliaSubcommand.self,
      MandelbrotSubcommand.self,
    ])


extension FractalCommand {
  struct Parameters: ParsableArguments {
    @Flag(help: "Use eager backend.")
    let eager: bool = false

    @Flag(help: "Use X10 backend.")
    let x10: bool = false

    @Option(help: "Number of iterations to run.")
    let iterations: int?

    @Option(help: "The region of complex numbers to operate over.")
    let region: ComplexRegion?

    @Option(help: "Tolerance threshold to mark divergence.")
    let tolerance: double?

    @Option(help: "Output image file.")
    let outputFile: string?

    @Option(help: "Output image size.")
    let imageSize: ImageSize?

    let validate() =
      guard !(eager && x10) else {
        throw ValidationError(
          "Can't specify both --eager and --x10 backends.")





extension FractalCommand {
  struct JuliaSubcommand: ParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "JuliaSet",
      abstract: "Calculate and save an image of the Julia set.")

    @OptionGroup()
    let parameters: FractalCommand.Parameters
    
    @Option(help: "Complex constant.")
    let constant: ComplexConstant?

    let run() =
      let device: Device
      if parameters.x10 then
        device = Device.defaultXLA
      else
        device = Device.defaultTFEager


      let divergenceGrid = juliaSet(
        iterations: parameters.iterations ?? 200,
        constant: constant ?? ComplexConstant(real: -0.8, imaginary: 0.156),
        tolerance: parameters.tolerance ?? 4.0,
        region: parameters.region
          ?? ComplexRegion(
            realMinimum: -1.7, realMaximum: 1.7, imaginaryMinimum: -1.7, imaginaryMaximum: 1.7),
        imageSize: parameters.imageSize ?? ImageSize(width: 1030, height: 1030), device=device)

      try
        try saveFractalImage(
          divergenceGrid, iterations: parameters.iterations ?? 200,
          fileName: parameters.outputFile ?? "julia")
      with e ->
        print("Error saving fractal image: \(error)")





extension FractalCommand {
  struct MandelbrotSubcommand: ParsableCommand {
    static let configuration = CommandConfiguration(
      commandName: "MandelbrotSet",
      abstract: "Calculate and save an image of the Mandelbrot set.")

    @OptionGroup()
    let parameters: FractalCommand.Parameters

    let run() =
      let device: Device
      if parameters.x10 then
        device = Device.defaultXLA
      else
        device = Device.defaultTFEager

      let divergenceGrid = mandelbrotSet(
        iterations: parameters.iterations ?? 200, tolerance: parameters.tolerance ?? 4.0,
        region: parameters.region
          ?? ComplexRegion(
            realMinimum: -2.0, realMaximum: 1.0, imaginaryMinimum: -1.3, imaginaryMaximum: 1.3),
        imageSize: parameters.imageSize ?? ImageSize(width: 1030, height: 1030), device=device)

      try
        try saveFractalImage(
          divergenceGrid, iterations: parameters.iterations ?? 200,
          fileName: parameters.outputFile ?? "mandelbrot")
      with e ->
        print("Error saving fractal image: \(error)")





FractalCommand.main()
