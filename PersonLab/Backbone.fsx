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

open Checkpoints
open DiffSharp

type DepthwiseSeparableConvBlock() =
  inherit Model()
  let dConv: DepthwiseConv2d<Float>
  let conv: Conv2d

  public init(
    depthWiseFilter: Tensor,
    depthWiseBias: Tensor,
    pointWiseFilter: Tensor,
    pointWiseBias: Tensor,
    strides = [Int, Int)
  ) = 

    dConv = DepthwiseConv2d(
      filter: depthWiseFilter,
      bias: depthWiseBias,
      activation=dsharp.relu6,
      strides=strides,
      padding=kernelSize/2 (* "same " *)
    )

    conv = Conv2d(
      filter: pointWiseFilter,
      bias: pointWiseBias,
      activation=dsharp.relu6,
      padding=kernelSize/2 (* "same " *)
    )


  
  override _.forward(input) =
    input |> dConv, conv)



type MobileNetLikeBackbone() =
  inherit Model()
  let ckpt: CheckpointReader

  let convBlock0: Conv2d
  let dConvBlock1: DepthwiseSeparableConvBlock
  let dConvBlock2: DepthwiseSeparableConvBlock
  let dConvBlock3: DepthwiseSeparableConvBlock
  let dConvBlock4: DepthwiseSeparableConvBlock
  let dConvBlock5: DepthwiseSeparableConvBlock
  let dConvBlock6: DepthwiseSeparableConvBlock
  let dConvBlock7: DepthwiseSeparableConvBlock
  let dConvBlock8: DepthwiseSeparableConvBlock
  let dConvBlock9: DepthwiseSeparableConvBlock
  let dConvBlock10: DepthwiseSeparableConvBlock
  let dConvBlock11: DepthwiseSeparableConvBlock
  let dConvBlock12: DepthwiseSeparableConvBlock
  let dConvBlock13: DepthwiseSeparableConvBlock

  public init(checkpoint: CheckpointReader) = 
    self.ckpt = checkpoint

    self.convBlock0 = Conv2d(
      filter: ckpt.load("Conv2d_0/weights"),
      bias: ckpt.load("Conv2d_0/biases"),
      activation=dsharp.relu6,
      stride=2,
      padding=kernelSize/2 (* "same " *)
    )
    self.dConvBlock1 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_1_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_1_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_1_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_1_pointwise/biases"),
      stride=1
    )
    self.dConvBlock2 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_2_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_2_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_2_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_2_pointwise/biases"),
      stride=2
    )
    self.dConvBlock3 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_3_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_3_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_3_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_3_pointwise/biases"),
      stride=1
    )
    self.dConvBlock4 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_4_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_4_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_4_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_4_pointwise/biases"),
      stride=2
    )
    self.dConvBlock5 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_5_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_5_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_5_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_5_pointwise/biases"),
      stride=1
    )
    self.dConvBlock6 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_6_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_6_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_6_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_6_pointwise/biases"),
      stride=2
    )
    self.dConvBlock7 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_7_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_7_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_7_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_7_pointwise/biases"),
      stride=1
    )
    self.dConvBlock8 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_8_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_8_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_8_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_8_pointwise/biases"),
      stride=1
    )
    self.dConvBlock9 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_9_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_9_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_9_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_9_pointwise/biases"),
      stride=1
    )
    self.dConvBlock10 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_10_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_10_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_10_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_10_pointwise/biases"),
      stride=1
    )
    self.dConvBlock11 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_11_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_11_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_11_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_11_pointwise/biases"),
      stride=1
    )
    self.dConvBlock12 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_12_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_12_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_12_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_12_pointwise/biases"),
      stride=1
    )
    self.dConvBlock13 = DepthwiseSeparableConvBlock(
      depthWiseFilter: ckpt.load("Conv2d_13_depthwise/depthwise_weights"),
      depthWiseBias: ckpt.load("Conv2d_13_depthwise/biases"),
      pointWiseFilter: ckpt.load("Conv2d_13_pointwise/weights"),
      pointWiseBias: ckpt.load("Conv2d_13_pointwise/biases"),
      stride=1
    )


  
  override _.forward(input) =
    let x = convBlock0(input)
    x = dConvBlock1(x)
    x = dConvBlock2(x)
    x = dConvBlock3(x)
    x = dConvBlock4(x)
    x = dConvBlock5(x)
    x = dConvBlock6(x)
    x = dConvBlock7(x)
    x = dConvBlock8(x)
    x = dConvBlock9(x)
    x = dConvBlock10(x)
    x = dConvBlock11(x)
    x = dConvBlock12(x)
    x = dConvBlock13(x)
    x



