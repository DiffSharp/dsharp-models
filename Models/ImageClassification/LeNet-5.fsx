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

open DiffSharp

// Original Paper:
// "Gradient-Based Learning Applied to Document Recognition"
// Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner
// http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
//
// Note: this implementation connects all the feature maps in the second convolutional layer.
// Additionally, ReLU is used instead of sigmoid activations.

type LeNet: Layer {
    let conv1 = Conv2d(filterShape=(5, 5, 1, 6), padding="same", activation= relu)
    let pool1 = AvgPool2D<Float>(poolSize: (2, 2), stride=2)
    let conv2 = Conv2d(filterShape=(5, 5, 6, 16), activation= relu)
    let pool2 = AvgPool2D<Float>(poolSize: (2, 2), stride=2)
    let flatten = Flatten()
    let fc1 = Dense(inputSize=400, outputSize=120, activation= relu)
    let fc2 = Dense(inputSize=120, outputSize=84, activation= relu)
    let fc3 = Dense(inputSize=84, outputSize=10)

    public init() =

    
    override _.forward(input) =
        let convolved = input |> conv1, pool1, conv2, pool2)
        return convolved |> flatten, fc1, fc2, fc3)


