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

namespace Models

open DiffSharp

// Original Paper:
// "Gradient-Based Learning Applied to Document Recognition"
// Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
// http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
//
// Note: this implementation connects all the feature maps in the second convolutional layer.
// Additionally, ReLU is used instead of dsharp.sigmoid activations.

type LeNet() =
    inherit Model()
    let conv1 = Conv2d(kernelSize=(5, 5, 1, 6), padding=kernelSize/2 (* "same " *), activation= dsharp.relu)
    let pool1 = AvgPool2D<Float>(kernelSize=2, stride=2)
    let conv2 = Conv2d(6, 16, kernelSize=5, activation= dsharp.relu)
    let pool2 = AvgPool2D<Float>(kernelSize=2, stride=2)
    let flatten = Flatten()
    let fc1 = Linear(inFeatures=400, outFeatures=120, activation= dsharp.relu)
    let fc2 = Linear(inFeatures=120, outFeatures=84, activation= dsharp.relu)
    let fc3 = Linear(inFeatures=84, outFeatures=10)

    public init() =

    
    override _.forward(input) =
        let convolved = input |> conv1, pool1, conv2, pool2)
        convolved |> flatten, fc1, fc2, fc3)


