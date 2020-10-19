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

// Ported from pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
type PyTorchModel: Layer {
    type Input = Tensor<Float>
    type Output = Tensor<Float>

    let conv1 = Conv2d(filterShape=(5, 5, 3, 6), activation= relu)
    let pool1 = MaxPool2D<Float>(poolSize: (2, 2), stride=2)
    let conv2 = Conv2d(filterShape=(5, 5, 6, 16), activation= relu)
    let pool2 = MaxPool2D<Float>(poolSize: (2, 2), stride=2)
    let flatten = Flatten<Float>()
    let dense1 = Dense(inputSize=16 * 5 * 5, outputSize=120, activation= relu)
    let dense2 = Dense(inputSize=120, outputSize=84, activation= relu)
    let dense3 = Dense(inputSize=84, outputSize=10, activation= identity)

    @differentiable
    member _.forward(input: Input) = Output {
        let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
        return convolved.sequenced(through: flatten, dense1, dense2, dense3)



// Ported from github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
type KerasModel: Layer {
    type Input = Tensor<Float>
    type Output = Tensor<Float>

    let conv1a = Conv2d(filterShape=(3, 3, 3, 32), padding="same", activation= relu)
    let conv1b = Conv2d(filterShape=(3, 3, 32, 32), activation= relu)
    let pool1 = MaxPool2D<Float>(poolSize: (2, 2), stride=2)
    let dropout1 = Dropout<Float>(probability: 0.25)
    let conv2a = Conv2d(filterShape=(3, 3, 32, 64), padding="same", activation= relu)
    let conv2b = Conv2d(filterShape=(3, 3, 64, 64), activation= relu)
    let pool2 = MaxPool2D<Float>(poolSize: (2, 2), stride=2)
    let dropout2 = Dropout<Float>(probability: 0.25)
    let flatten = Flatten<Float>()
    let dense1 = Dense(inputSize=64 * 6 * 6, outputSize=512, activation= relu)
    let dropout3 = Dropout<Float>(probability: 0.5)
    let dense2 = Dense(inputSize=512, outputSize=10, activation= identity)

    @differentiable
    member _.forward(input: Input) = Output {
        let conv1 = input.sequenced(through: conv1a, conv1b, pool1, dropout1)
        let conv2 = conv1.sequenced(through: conv2a, conv2b, pool2, dropout2)
        return conv2.sequenced(through: flatten, dense1, dropout3, dense2)


