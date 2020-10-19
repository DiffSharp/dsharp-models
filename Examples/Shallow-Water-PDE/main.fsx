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
open Benchmark


open DiffSharp

// MARK: Command line interface

type ShallowWaterPDE: ParsableCommand {
  static let configuration = CommandConfiguration(
    discussion: "Solve shallow water PDE on a unit square."
  )

  enum Task: string, EnumerableFlag {
    case splash, optimization, benchmark

  enum CodingKeys: string, CodingKey {
    case tasks

  @Flag(help: "Task to run.")
  let tasks: [Task] = [.splash]

  let n = 256
  let duration = 512

  /// Runs a simple simulation in a rectangular bathtub initialized with Dirac delta function.
  let runSplash() = 
    let initialSplashLevel = Tensor<Float>(zeros: [n, n])
    initialSplashLevel[n / 2, n / 2] = dsharp.tensor(100)

    let initialSplash = TensorSliceSolution(waterLevel: initialSplashLevel)
    let splashEvolution = [TensorSliceSolution](evolve: initialSplash, duration)

    for (i, solution) in splashEvolution.enumerated() = 
      let file = Uri(fileURLWithPath= "Images/Splash-\(String(format: "%03d", i)).jpg")
      solution.visualization.waterLevel.save(file, format: .grayscale, quality: 100)



  /// Runs an optimization through time-steps and updates the initial water height to obtain a specific wave patter at the end.
  let runOptimization() = 
    let α: double = 500.0
    let initialWaterLevel = Tensor<Float>(zeros: [n, n])

    let targetImage = Image(jpeg: Uri(fileURLWithPath= "Images/Target.jpg"))
    let target = targetImage.tensor - double(byte.max) / 2
    target = target.mean(squeezingAxes: 2) / double(byte.max)

    for opt in 1...200 {

      let (loss, del_initialWaterLevel) = valueWithGradient(at: initialWaterLevel) = 
        (initialWaterLevel) = Float in
        let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
        let evolution = [TensorSliceSolution](evolve: initialSolution, duration)

        let last = withoutDerivative(at: evolution.count - 1)
        let loss = evolution[last].meanSquaredError(target)
        return loss


      print("\(opt): \(loss)")
      initialWaterLevel.move(along: del_initialWaterLevel.scaled(by: -α))


    let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
    let evolution = [TensorSliceSolution](evolve: initialSolution, duration)

    for (i, solution) in evolution.enumerated() = 
      let file = Uri(fileURLWithPath= "Images/Optimization-\(String(format: "%03d", i)).jpg")
      solution.visualization.waterLevel.save(file, format: .grayscale, quality: 100)



  private let runSplashArrayLoopBenchmark() = 
    let initialWaterLevel = double[][](repeating: double[](repeating: 0.0, count: n), count: n)
    initialWaterLevel[n / 2][n / 2] = 100

    let initialSolution = ArrayLoopSolution(waterLevel: initialWaterLevel)
    _ = [ArrayLoopSolution](evolve: initialSolution, duration)


  private let runSplashTensorLoopBenchmark(on device: Device) = 
    let initialWaterLevel = Tensor<Float>(zeros: [n, n], device=device)
    initialWaterLevel[n / 2][n / 2] = Tensor<Float>(100, device=device)

    let initialSolution = TensorLoopSolution(waterLevel: initialWaterLevel)
    _ = [TensorLoopSolution](evolve: initialSolution, duration)


  private let runSplashTensorSliceBenchmark(on device: Device) = 
    let initialWaterLevel = Tensor<Float>(zeros: [n, n], device=device)
    initialWaterLevel[n / 2][n / 2] = Tensor<Float>(100, device=device)

    let initialSolution = TensorSliceSolution(waterLevel: initialWaterLevel)
    _ = [TensorSliceSolution](evolve: initialSolution, duration)


  private let runSplashTensorConvBenchmark(on device: Device) = 
    let initialWaterLevel = Tensor<Float>(zeros: [n, n], device=device)
    initialWaterLevel[n / 2][n / 2] = Tensor<Float>(100, device=device)

    let initialSolution = TensorConvSolution(waterLevel: initialWaterLevel)
    _ = [TensorConvSolution](evolve: initialSolution, duration)


  /// Benchmark suite that exercises the 3 different solver implementations on a simple problem without back-propagation.
  let splashBenchmarks: BenchmarkSuite {
    BenchmarkSuite(
      name= "Shallow Water PDE Solver",
      settings: Iterations(10), WarmupIterations(2)
    ) =  suite in
      suite.benchmark("Array Loop") = 
        runSplashArrayLoopBenchmark()


      //            FIXME: This is at least 1000x slower. One can easily grow old while waiting... :(
      //            suite.benchmark("Tensor Loop") = 
      //                runSplashTensorLoopBenchmark(on: Device.default)
      //
      //            suite.benchmark("Tensor Loop (XLA)") = 
      //                runSplashTensorLoopBenchmark(on: Device.defaultXLA)
      //

      suite.benchmark("Tensor Slice") = 
        runSplashTensorSliceBenchmark(on: Device.default)

      suite.benchmark("Tensor Slice (XLA)") = 
        runSplashTensorSliceBenchmark(on: Device.defaultXLA)


      suite.benchmark("Tensor Conv") = 
        runSplashTensorConvBenchmark(on: Device.default)

      suite.benchmark("Tensor Conv (XLA)") = 
        runSplashTensorConvBenchmark(on: Device.defaultXLA)




  mutating let run() =
    for task in tasks {
      match task with
      | .splash ->
        runSplash()
      | .optimization ->
        runOptimization()
      | .benchmark ->
        let runner = BenchmarkRunner(
          suites: [splashBenchmarks], settings: [TimeUnit(.ms)], customDefaults: [])
        try runner.run()





// MARK: - Main

ShallowWaterPDE.main()
