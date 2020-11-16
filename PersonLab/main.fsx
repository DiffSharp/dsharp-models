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

#r @"..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\bin\Debug\netcoreapp3.1\publish\Library.dll"


open DiffSharp

type Inference: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "personlab",
    abstract: """
      Runs human pose estimation on a local image file.
      """
  )

  @Argument(help: "Path to local image to run pose estimation on")
  let imagePath: string

  @Option(name= .shortAndLong, help: "Path to checkpoint directory")
  let checkpointPath: string?

  @Flag(name= .shortAndLong, help: "Print profiling data")
  let profiling: bool

  let run() = 
    model.mode <- Mode.Eval
    let config = Config(printProfilingData: profiling)
    if checkpointPath <> nil then
      config.checkpointPath = Uri(fileURLWithPath= checkpointPath!)

    let model = PersonLab(config)

    let fileManager = FileManager()
    if not (fileManager.Exists(imagePath)) then
      print($"No image found at path: {imagePath}")
      return

    let image = Image(jpeg: Uri(fileURLWithPath= imagePath))

    let poses = [Pose]()
    if profiling then
      print("Running model 10 times to see how inference time changes.")
      for _ in 1...10 do
        poses = model(image)

    else
      poses = model(image)


    let drawnTensor = image.tensor
    for pose in poses do
      draw(pose, device=&drawnTensor)

    drawnTensor.saveImage("out.jpg")
    print("Output image saved to 'out.jpg'")



Inference.main()
