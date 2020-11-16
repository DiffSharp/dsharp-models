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
    let pool1 = MaxPool2d(kernelSize=2, stride=2)
    let conv2 = Conv2d(filterShape=(5, 5, 6, 16), activation= relu)
    let pool2 = MaxPool2d(kernelSize=2, stride=2)
    let flatten = Flatten()
    let dense1 = Linear(inFeatures=16 * 5 * 5, outFeatures=120, activation= relu)
    let dense2 = Linear(inFeatures=120, outFeatures=84, activation= relu)
    let dense3 = Linear(inFeatures=84, outFeatures=10, activation= identity)

    
    override _.forward(input: Input) = Output {
        let convolved = input |> conv1, pool1, conv2, pool2)
        return convolved |> flatten, dense1, dense2, dense3)



// Ported from github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
type KerasModel: Layer {
    type Input = Tensor<Float>
    type Output = Tensor<Float>

    let conv1a = Conv2d(filterShape=(3, 3, 3, 32), padding="same", activation= relu)
    let conv1b = Conv2d(filterShape=(3, 3, 32, 32), activation= relu)
    let pool1 = MaxPool2d(kernelSize=2, stride=2)
    let dropout1 = Dropout2d(p=0.25)
    let conv2a = Conv2d(filterShape=(3, 3, 32, 64), padding="same", activation= relu)
    let conv2b = Conv2d(filterShape=(3, 3, 64, 64), activation= relu)
    let pool2 = MaxPool2d(kernelSize=2, stride=2)
    let dropout2 = Dropout2d(p=0.25)
    let flatten = Flatten()
    let dense1 = Linear(inFeatures=64 * 6 * 6, outFeatures=512, activation= relu)
    let dropout3 = Dropout2d(p=0.5)
    let dense2 = Linear(inFeatures=512, outFeatures=10, activation= identity)

    
    override _.forward(input: Input) = Output {
        let conv1 = input |> conv1a, conv1b, pool1, dropout1)
        let conv2 = conv1 |> conv2a, conv2b, pool2, dropout2)
        return conv2 |> flatten, dense1, dropout3, dense2)


