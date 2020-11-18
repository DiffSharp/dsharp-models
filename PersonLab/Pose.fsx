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

type Keypoint {
  let y: double
  let x: double
  let index: KeypointIndex
  let score: double

  init(
    heatmapY: int, heatmapX: int, index: int, score: double, offsets: CPUTensor<Float>,
    outputStride: int
  ) = 
    self.y = double(heatmapY) * double(outputStride) + offsets[heatmapY, heatmapX, index]
    self.x =
      double(heatmapX) * double(outputStride)
      + offsets[heatmapY, heatmapX, index + KeypointIndex.allCases.count]
    self.index = KeypointIndex(rawValue: index)!
    self.score = score


  init(y: double, x: double, index: KeypointIndex, score: double) = 
    self.y = y
    self.x = x
    self.index = index
    self.score = score


  let isWithinRadiusOfCorrespondingKeypoints(in poses: Pose[], radius: double) = Bool {
    poses.contains { pose in
      let correspondingKeypoint = pose.getKeypoint(self.index)!
      let dy = correspondingKeypoint.y - self.y
      let dx = correspondingKeypoint.x - self.x
      let squaredDistance = dy * dy + dx * dx
      squaredDistance <= radius * radius




type KeypointIndex: int, CaseIterable {
  case nose = 0
  case leftEye
  case rightEye
  case leftEar
  case rightEar
  case leftShoulder
  case rightShoulder
  case leftElbow
  case rightElbow
  case leftWrist
  case rightWrist
  case leftHip
  case rightHip
  case leftKnee
  case rightKnee
  case leftAnkle
  case rightAnkle


type Direction { case fwd, bwd

let getNextKeypointIndexAndDirection(keypointId: KeypointIndex) = [(KeypointIndex, Direction)] {
  match keypointId with
  | .nose ->
    [(.leftEye, .fwd), (.rightEye, .fwd), (.leftShoulder, .fwd), (.rightShoulder, .fwd)]
  | .leftEye -> return [(.nose, .bwd), (.leftEar, .fwd)]
  | .rightEye -> return [(.nose, .bwd), (.rightEar, .fwd)]
  | .leftEar -> return [(.leftEye, .bwd)]
  | .rightEar -> return [(.rightEye, .bwd)]
  | .leftShoulder -> return [(.leftHip, .fwd), (.leftElbow, .fwd), (.nose, .bwd)]
  | .rightShoulder -> return [(.rightHip, .fwd), (.rightElbow, .fwd), (.nose, .bwd)]
  | .leftElbow -> return [(.leftWrist, .fwd), (.leftShoulder, .bwd)]
  | .rightElbow -> return [(.rightWrist, .fwd), (.rightShoulder, .bwd)]
  | .leftWrist -> return [(.leftElbow, .bwd)]
  | .rightWrist -> return [(.rightElbow, .bwd)]
  | .leftHip -> return [(.leftKnee, .fwd), (.leftShoulder, .bwd)]
  | .rightHip -> return [(.rightKnee, .fwd), (.rightShoulder, .bwd)]
  | .leftKnee -> return [(.leftAnkle, .fwd), (.leftHip, .bwd)]
  | .rightKnee -> return [(.rightAnkle, .fwd), (.rightHip, .bwd)]
  | .leftAnkle -> return [(.leftKnee, .bwd)]
  | .rightAnkle -> return [(.rightKnee, .bwd)]



/// Maps a pair of keypoint indexes to the appropiate index to be used
/// in the displacement forward and backward tensors.
let keypointPairToDisplacementIndexMap: [Set<KeypointIndex>: int] = [
  Set([.nose, .leftEye]): 0,
  Set([.leftEye, .leftEar]): 1,
  Set([.nose, .rightEye]): 2,
  Set([.rightEye, .rightEar]): 3,
  Set([.nose, .leftShoulder]): 4,
  Set([.leftShoulder, .leftElbow]): 5,
  Set([.leftElbow, .leftWrist]): 6,
  Set([.leftShoulder, .leftHip]): 7,
  Set([.leftHip, .leftKnee]): 8,
  Set([.leftKnee, .leftAnkle]): 9,
  Set([.nose, .rightShoulder]): 10,
  Set([.rightShoulder, .rightElbow]): 11,
  Set([.rightElbow, .rightWrist]): 12,
  Set([.rightShoulder, .rightHip]): 13,
  Set([.rightHip, .rightKnee]): 14,
  Set([.rightKnee, .rightAnkle]): 15,
]

type Pose {
  let keypoints: [Keypoint?] = Array.replicate nil, count: KeypointIndex.allCases.count)
  let resolution: (height: int, width: int)

  mutating let add(keypoint: Keypoint) = 
    keypoints[keypoint.index.rawValue] = keypoint


  let getKeypoint(index: KeypointIndex) = Keypoint? {
    keypoints[index.rawValue]


  mutating let rescale(to newResolution: (height: int, width: int)) = 
    for i in 0..<keypoints.count do
      if let k = keypoints[i] then
        k.y *= double(newResolution.height) / double(resolution.height)
        k.x *= double(newResolution.width) / double(resolution.width)
        self.keypoints[i] = k


    self.resolution = newResolution



extension Pose: CustomStringConvertible {
  let description: string {
    let description = ""
    for keypoint in keypoints do
      description.append(
        $"\(keypoint!.index) - \(keypoint!.score) | \(keypoint!.y) - \(keypoint!.x)\n")

    description


