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
// "Learning Spatiotemporal Features with 3D Convolutional Networks"
// Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, Manohar Paluri
// https://arxiv.org/pdf/1412.0767.pdf

type C3D: Layer {
    
    // Model presumes input of TensorShape([1, 12, 256, 256, 3])
    
    let conv1 = Conv3D<Float>(filterShape=(3, 3, 3, 3, 32), activation= relu)
    let conv2 = Conv3D<Float>(filterShape=(3, 3, 3, 32, 64), activation= relu)
    let conv3 = Conv3D<Float>(filterShape=(3, 3, 3, 64, 128), activation= relu)
    let conv4 = Conv3D<Float>(filterShape=(3, 3, 3, 128, 128), activation= relu)
    let conv5 = Conv3D<Float>(filterShape=(2, 2, 2, 128, 256), activation= relu)
    let conv6 = Conv3D<Float>(filterShape=(2, 2, 2, 256, 256), activation= relu)
    
    let pool = MaxPool3D<Float>(poolSize: (1, 2, 2), strides = [1, 2, 2))
    let flatten = Flatten<Float>()
    let dropout = Dropout<Float>(probability: 0.5)
    
    let dense1 = Dense(inputSize=86528, outputSize=1024)
    let dense2 = Dense(inputSize=1024, outputSize=1024)
    let output: Dense
    
    public init(classCount: int) = 
        self.output = Dense(inputSize=1024, outputSize=classCount)

    
    @differentiable
    member _.forward(input: Tensor) : Tensor (* <Float> *) {
        return input
            .sequenced(through: conv1, pool, conv2, pool)
            .sequenced(through: conv3, conv4, pool, conv5, conv6, pool)
            .sequenced(through: flatten, dense1, dropout, dense2, output)


