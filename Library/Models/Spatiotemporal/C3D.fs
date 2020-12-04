// Copyright 2019 The TensorFlow Authors, adapted by the DiffSharp authors. All Rights Reserved.
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

module Models.ImageClassification.C3D

open DiffSharp
open DiffSharp.Model

// Original Paper:
// "Learning Spatiotemporal Features with 3D Convolutional Networks"
// Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, Manohar Paluri
// https://arxiv.org/pdf/1412.0767.pdf

type C3D(classCount: int) =
    inherit Model()
    
    // Model presumes input of [[1, 12, 256, 256, 3])
    
    let conv1 = Conv3d(3, 32, kernelSize=3) --> dsharp.relu
    let conv2 = Conv3d(32, 64, kernelSize=3) --> dsharp.relu
    let conv3 = Conv3d(64, 128, kernelSize=3) --> dsharp.relu
    let conv4 = Conv3d(128, 128, kernelSize=3) --> dsharp.relu
    let conv5 = Conv3d(128, 256, kernelSize=2) --> dsharp.relu
    let conv6 = Conv3d(256, 256, kernelSize=2) --> dsharp.relu
    
    let pool = MaxPool3d(kernelSizes=[1;2;2], strides = [1; 2; 2])
    let flatten = Flatten()
    let dropout = Dropout2d(p=0.5)
    
    let dense1 = Linear(inFeatures=86528, outFeatures=1024)
    let dense2 = Linear(inFeatures=1024, outFeatures=1024)
    let output = Linear(inFeatures=1024, outFeatures=classCount)
    
    override _.forward(input) =
        input
        |> conv1.forward |> pool.forward |> conv2.forward |> pool.forward
        |> conv3.forward |> conv4.forward |> pool.forward |> conv5.forward |> conv6.forward |> pool.forward
        |> flatten.forward |> dense1.forward |> dropout.forward |> dense2.forward |> output.forward
