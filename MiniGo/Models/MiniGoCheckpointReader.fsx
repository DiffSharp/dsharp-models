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

open Checkpoints
open DiffSharp

public class MiniGoCheckpointReader: CheckpointReader {
    let layerCounts: Map<string, int> = [:]

    let readTensor(layerName: string, weightName: string) = Tensor<Float>? =
        let countSuffix = layerCounts[layerName] = nil ? "" : "_\(layerCounts[layerName]!)"
        let tensorName = layerName + countSuffix + "/" + weightName
        guard containsTensor(named: tensorName) else { return nil
        Tensor<Float>(loadTensor(named: tensorName))


    /// Increments a per-layer counter for variable names in the checkpoint file.
    /// As the Python model code uses low-level TensorFlow APIs, variables are namespaced only by
    /// layer name and this per-layer counter (e.g., conv2d_5/bias).
    let increment(layerName: string) = 
        layerCounts[layerName, | _ -> 0] += 1



let checkShapes(tensor1: Tensor, _ tensor2: Tensor) = 
    guard tensor1.shape = tensor2.shape else {
        print($"Shape mismatch: {tensor1.shape} <> {tensor2.shape}")
        fatalError()



type ILoadableFromPythonCheckpoint {
    mutating let load(from reader: MiniGoCheckpointReader)


extension Dense: LoadableFromPythonCheckpoint where Scalar = Float {
    mutating let load(from reader: MiniGoCheckpointReader) = 
        let newWeight = reader.readTensor(layerName: "dense", weightName: "kernel")!
        checkShapes(weight, newWeight)
        weight = newWeight

        if let newBias = reader.readTensor(layerName: "dense", weightName: "bias") then
            checkShapes(bias, newBias)
            bias = newBias

        reader.increment(layerName: "dense")



extension Conv2D: LoadableFromPythonCheckpoint where Scalar = Float {
    mutating let load(from reader: MiniGoCheckpointReader) = 
        let newFilter = reader.readTensor(layerName: "conv2d", weightName: "kernel")!
        checkShapes(filter, newFilter)
        filter = newFilter

        // TODO(jekbradbury): handle layers with optional weights
        // It would be helpful to have an op to see if a checkpoint contains a particular variable
        // (see b/124126672)
        // if let newBias = loader.readTensor(layerName: "conv2d", weightName: "bias") then
        //   checkShapes(bias, newBias)
        //   bias = newBias
        //

        reader.increment(layerName: "conv2d")



extension BatchNorm: LoadableFromPythonCheckpoint where Scalar = Float {
    mutating let load(from reader: MiniGoCheckpointReader) = 
        if let newOffset = reader.readTensor(layerName: "batch_normalization", weightName: "beta") then
            checkShapes(offset, newOffset)
            offset = newOffset


        if let newScale = reader.readTensor(layerName: "batch_normalization", weightName: "gamma") then
            checkShapes(scale, newScale)
            scale = newScale


        if let newRunningMean = reader.readTensor(
            layerName: "batch_normalization",
            weightName: "moving_mean") = 
            // Do not check shapes, because running mean/variance are initialized to scalar
            // tensors.
            runningMean.value = newRunningMean


        if let newRunningVariance = reader.readTensor(
            layerName: "batch_normalization",
            weightName: "moving_variance") = 
            // Do not check shapes, because running mean/variance are initialized to scalar
            // tensors.
            runningVariance.value = newRunningVariance


        reader.increment(layerName: "batch_normalization")


