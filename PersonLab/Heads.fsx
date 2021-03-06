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

#r @"..\bin\Debug\net5.0\publish\DiffSharp.Core.dll"
#r @"..\bin\Debug\net5.0\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\bin\Debug\net5.0\publish\Library.dll"

open Checkpoints
open DiffSharp

type PersonlabHeadsResults: Differentiable {
  let heatmap: Tensor
  let offsets: Tensor
  let displacementsFwd: Tensor
  let displacementsBwd: Tensor


type PersonlabHeads() =
  inherit Model()
  let ckpt: CheckpointReader

  let heatmap: Conv2d
  let offsets: Conv2d
  let displacementsFwd: Conv2d
  let displacementsBwd: Conv2d

  public init(checkpoint: CheckpointReader) = 
    self.ckpt = checkpoint

    self.heatmap = Conv2d(
      filter: ckpt.load("heatmap_2/weights"),
      bias: ckpt.load("heatmap_2/biases"),
      padding=kernelSize/2 (* "same " *)
    )
    self.offsets = Conv2d(
      filter: ckpt.load("offset_2/weights"),
      bias: ckpt.load("offset_2/biases"),
      padding=kernelSize/2 (* "same " *)
    )
    self.displacementsFwd = Conv2d(
      filter: ckpt.load("displacement_fwd_2/weights"),
      bias: ckpt.load("displacement_fwd_2/biases"),
      padding=kernelSize/2 (* "same " *)
    )
    self.displacementsBwd = Conv2d(
      filter: ckpt.load("displacement_bwd_2/weights"),
      bias: ckpt.load("displacement_bwd_2/biases"),
      padding=kernelSize/2 (* "same " *)
    )


  
  override _.forward(input: Tensor) = PersonlabHeadsResults {
    PersonlabHeadsResults(
      heatmap: dsharp.sigmoid(self.heatmap(input)),
      offsets: self.offsets(input),
      displacementsFwd: self.displacementsFwd(input),
      displacementsBwd: self.displacementsBwd(input)
    )


