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

#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"

open DiffSharp
open DiffSharp.Model

// Ported from pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
type PyTorchModel() = 
    inherit Model()

    let conv1 = Conv2d(3, 6, 5, activation=dsharp.relu)
    let pool1 = MaxPool2d(kernelSize=2, stride=2)
    let conv2 = Conv2d(6, 16, kernelSize=5, activation=dsharp.relu)
    let pool2 = MaxPool2d(kernelSize=2, stride=2)
    let flatten = Flatten()
    let dense1 = Linear(inFeatures=16 * 5 * 5, outFeatures=120, activation=dsharp.relu)
    let dense2 = Linear(inFeatures=120, outFeatures=84, activation=dsharp.relu)
    let dense3 = Linear(inFeatures=84, outFeatures=10, activation= id)
    
    override _.forward(input: Tensor) =
        let convolved = input |> conv1.forward |> pool1.forward |> conv2.forward |> pool2.forward
        convolved |> flatten.forward |> dense1.forward |> dense2.forward |> dense3.forward

// Ported from github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
type KerasModel() = 
    inherit Model()

    let conv1a = Conv2d(4, 32, kernelSize=3, padding=3/2 (* "same" *) , activation=dsharp.relu)
    let conv1b = Conv2d(32, 32, kernelSize=3, activation=dsharp.relu)
    let pool1 = MaxPool2d(kernelSize=2, stride=2)
    let dropout1 = Dropout2d(p=0.25)
    let conv2a = Conv2d(32, 64, kernelSize=3, padding=3/2 (* "same" *), activation=dsharp.relu)
    let conv2b = Conv2d(64, 64, kernelSize=3, activation=dsharp.relu)
    let pool2 = MaxPool2d(kernelSize=2, stride=2)
    let dropout2 = Dropout2d(p=0.25)
    let flatten = Flatten()
    let dense1 = Linear(inFeatures=64 * 6 * 6, outFeatures=512, activation=dsharp.relu)
    let dropout3 = Dropout2d(p=0.5)
    let dense2 = Linear(inFeatures=512, outFeatures=10, activation= id)

    override _.forward(input: Tensor) =
        let conv1 = input |> conv1a.forward |> conv1b.forward |> pool1.forward |> dropout1.forward
        let conv2 = conv1 |> conv2a.forward |> conv2b.forward |> pool2.forward |> dropout2.forward
        conv2 |> flatten.forward |> dense1.forward |> dropout3.forward |> dense2.forward


