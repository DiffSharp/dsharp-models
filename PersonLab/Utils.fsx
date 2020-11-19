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

open Checkpoints


open DiffSharp

type Config {
  let printProfilingData: bool
  let checkpointPath = Uri(
    string:
      "https://github.com/tryolabs/dsharp-models/releases/download/PersonlabDemo/personlabCheckpoint.zip"
  )!
  let inputImageSize = (height: 241, width: 289)

  // Decoder
  let outputStride = 16
  let poseScoreThreshold: double = 0.15
  let keypointScoreThreshold: double = 0.1
  let nmsRadius: double = 20.0
  let keypointLocalMaximumRadius = 1


extension CheckpointReader {
  let load(from name: string) : Tensor =
    dsharp.tensor(self.loadTensor(named: "MobilenetV1/{name}"))



let draw(pose: Pose, on imageTensor: inout Tensor) = 
  let pose = pose
  pose.rescale((height: imageTensor.shape.[0], width: imageTensor.shape.[1]))

  let recursivellyDrawNextKeypoint(
    after previousKeypoint: Keypoint, into imageTensor: inout Tensor
  ) = 
    for (nextKeypointIndex, direction) in getNextKeypointIndexAndDirection(previousKeypoint.index) do      if direction = .fwd then
        if let nextKeypoint = pose.getKeypoint(nextKeypointIndex) then
          drawLine(
            on: &imageTensor,
            from: (int(previousKeypoint.x), int(previousKeypoint.y)),
            (int(nextKeypoint.x), int(nextKeypoint.y))
          )
          recursivellyDrawNextKeypoint(after: nextKeypoint, into: &imageTensor)





  recursivellyDrawNextKeypoint(after: pose.getKeypoint(.nose)!, into: &imageTensor)


/// Used as an ad-hoc "hash" for tensor checking when copying the backbone from
/// our Python Tensorflow 1.5 version
let hash(tensor: Tensor) = 
  print(
    "[\(tensor.flattened().sum()), \(tensor[0, 0, 0]) \(tensor[0, -1, 1]), \(tensor[0, 1, 0]), \(tensor[0, -1, -1])]"
  )


/// Wrapper for Tensor which allows several order of magnitude faster subscript access,
/// as it avoids unnecesary GPU->CPU copies on each access.
type CPUTensor<T: TensorFlowScalar> {
  let flattenedTensor: T[]
  let shape: Shape

  init(tensor: Tensor<T>) = 
    self.flattenedTensor = tensor.scalars
    self.shape = tensor.shape


  subscript(indexes: int..) = T {
    let oneDimensionalIndex = 0
    for i in 1..ndims-1 do
      oneDimensionalIndex <- oneDimensionalIndex + indexes[i - 1] * shape.[i..].reduce(1, *)

    // Last dimension doesn't have multipliers.
    oneDimensionalIndex <- oneDimensionalIndex + indexes |> Array.last
    flattenedTensor[oneDimensionalIndex]


