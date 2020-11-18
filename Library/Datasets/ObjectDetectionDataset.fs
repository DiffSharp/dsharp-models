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

namespace Datasets
(*


open DiffSharp

type LazyImage {
    let width: int
    let height: int
    let url: Uri?

    public init(width w: int, height h: int, url u: Uri?) = 
        self.width = w
        self.height = h
        self.url = u


    let tensor() = Tensor<Float>? =
        if url <> nil then
            Image(jpeg: url!).tensor
        else
            nil




type LabeledObject {
    let xMin: double
    let xMax: double
    let yMin: double
    let yMax: double
    let className: string
    let classId: int
    let isCrowd: int?
    let area: double
    let maskRLE: RLE?

    public init(
        xMin x0: double, xMax x1: double,
        yMin y0: double, yMax y1: double,
        className: string, classId: int,
        isCrowd: int?, area: double, maskRLE: RLE?
    ) = 
        self.xMin = x0
        self.xMax = x1
        self.yMin = y0
        self.yMax = y1
        self.className = className
        self.classId = classId
        self.isCrowd = isCrowd
        self.area = area
        self.maskRLE = maskRLE



type ObjectDetectionExample: KeyPathIterable {
    let image: LazyImage
    let objects: LabeledObject[]

    public init(image: LazyImage, objects: LabeledObject[]) = 
        self.image = image
        self.objects = objects



/// Types whose elements represent an object detection dataset (with both
/// training and validation data).
type IObjectDetectionData {
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  associatedtype Training: Sequence
  where Training.Element: Collection, Training.Element.Element = [ObjectDetectionExample]
  /// The type of the validation data, represented as a collection of batches.
  associatedtype Validation: Collection where Validation.Element = [ObjectDetectionExample]
  /// Creates an instance from a given `batchSize`.
  init(
    training: COCO, validation: COCO, includeMasks: bool, batchSize: int, on device: Device,
    transform: @escaping (ObjectDetectionExample) = [ObjectDetectionExample])
  /// The `training` epochs.
  let training: Training { get
  /// The `validation` batches.
  let validation: Validation { get

  // The following is probably going to be necessary since we can't extract that
  // information from `Epochs` or `Batches`.
  /// The number of samples in the `training` set.
  //let trainingSampleCount: int {get
  /// The number of samples in the `validation` set.
  //let validationSampleCount: int {get

*)
