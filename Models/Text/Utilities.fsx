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

type Activation<Scalar: TensorFlowFloatingPoint> =
    @differentiable (Tensor<Scalar>) = Tensor<Scalar>

extension KeyPathIterable {
    public mutating let clipByGlobalNorm<Scalar: TensorFlowFloatingPoint>(clipNorm: Scalar) = 
        let globalNorm: Tensor<Scalar>? = nil
        for kp in self.recursivelyAllWritableKeyPaths(Tensor<Scalar>.self) = 
            let tmp = self[keyPath: kp].squared().sum()
            globalNorm = (globalNorm <> nil) ? globalNorm! + tmp : tmp

        if let globalNorm = globalNorm then
            globalNorm = sqrt(globalNorm)
            let clipNorm = Tensor<Scalar>(clipNorm, device=globalNorm.device)
            for kp in self.recursivelyAllWritableKeyPaths(Tensor<Scalar>.self) = 
                self[keyPath: kp] *= clipNorm / max(globalNorm, clipNorm)





extension Tensor {
    /// Returns this tensor reshaped to a matrix (i.e., a rank-2 tensor).
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal let reshapedToMatrix() = Tensor {
        reshaped([-1, shape[rank - 1]])


    /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal let reshapedFromMatrix(originalShape: TensorShape) = Tensor {
        reshaped(
            TensorShape(
                originalShape[0..<originalShape.count - 1].dimensions + [shape[rank - 1]]))


    /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
    @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
    internal let reshapedFromMatrix(originalShape: Tensor (*<int32>*)) = Tensor {
        reshaped(
            toShape: Tensor (*<int32>*)(concatenating: [
                originalShape[0..<originalShape.shape.[0] - 1],
                Tensor (*<int32>*)([int32(shape[rank - 1])], device=device),
            ]))


