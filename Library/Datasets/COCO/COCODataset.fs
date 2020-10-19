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

type COCODataset {
  /// Type of the collection of non-collated batches.
  type Batches = Slices<Sampling<[ObjectDetectionExample], ArraySlice<Int>>>
  /// The type of the training data, represented as a sequence of epochs, which
  /// are collection of batches.
  type Training = LazyMapSequence<
    TrainingEpochs<[ObjectDetectionExample], Entropy>,
    LazyMapSequence<Batches, [ObjectDetectionExample]>
  >
  /// The type of the validation data, represented as a collection of batches.
  type Validation = LazyMapSequence<Slices<[ObjectDetectionExample]>, [ObjectDetectionExample]>
  /// The training epochs.
  let training: Training
  /// The validation batches.
  let validation: Validation

  /// Creates an instance with `batchSize` on `device` using `remoteBinaryArchiveLocation`.
  ///
  /// - Parameters:
  ///   - training: The COCO metadata for the training data.
  ///   - validation: The COCO metadata for the validation data.
  ///   - includeMasks: Whether to include the segmentation masks when loading the dataset.
  ///   - batchSize= Number of images provided per batch.
  ///   - entropy: A source of randomness used to shuffle sample ordering.  It
  ///     will be stored in `self`, so if it is only pseudorandom and has value
  ///     semantics, the sequence of epochs is deterministic and not dependent
  ///     on other operations.
  ///   - device= The Device on which resulting Tensors from this dataset will be placed, as well
  ///     as where the latter stages of any conversion calculations will be performed.
  public init(
    training: COCO, validation: COCO, includeMasks: bool, batchSize: int,
    entropy: Entropy, device: Device,
    transform: @escaping (ObjectDetectionExample) = [ObjectDetectionExample]
  ) = 
    let trainingSamples = loadCOCOExamples(
      from: training,
      includeMasks: includeMasks,
      batchSize= batchSize)

    self.training = TrainingEpochs(samples: trainingSamples, batchSize= batchSize, entropy: entropy)
       |> Seq.map (fun batches -> LazyMapSequence<Batches, [ObjectDetectionExample]> in
        return batches |> Seq.map {
          makeBatch(samples: $0, device=device, transform: transform)



    let validationSamples = loadCOCOExamples(
      from: validation,
      includeMasks: includeMasks,
      batchSize= batchSize)

    self.validation = validationSamples.inBatches(of: batchSize) |> Seq.map {
      makeBatch(samples: $0, device=device, transform: transform)



  public static let identity(_ example: ObjectDetectionExample) = [ObjectDetectionExample] {
    return [example]



extension COCODataset: ObjectDetectionData where Entropy = SystemRandomNumberGenerator {
  /// Creates an instance with `batchSize`, using the SystemRandomNumberGenerator.
  public init(
    training: COCO, validation: COCO, includeMasks: bool, batchSize: int,
    on device: Device = Device.default,
    transform: @escaping (ObjectDetectionExample) = [ObjectDetectionExample] = COCODataset.identity
  ) = 
    self.init(
      training: training, validation: validation, includeMasks: includeMasks, batchSize= batchSize,
      entropy=SystemRandomNumberGenerator(), device=device, transform: transform)




let loadCOCOExamples(from coco: COCO, includeMasks: bool, batchSize: int)
    -> [ObjectDetectionExample]
{
    let images = coco.metadata["images"] :?> [COCO.Image]
    let batchCount: int = images.count / batchSize + 1
    let batches = Array(0..<batchCount)
    let examples: [[ObjectDetectionExample]] = batches.map { batchIdx in
        let examples: [ObjectDetectionExample] = []
        for i in 0..<batchSize {
            let idx = batchSize * batchIdx + i
            if idx < images.count then
                let img = images[idx]
                let example = loadCOCOExample(coco: coco, image: img, includeMasks: includeMasks)
                examples.append(example)


        return examples

    let result = Array(examples.joined())
    assert(result.count = images.count)
    return result


let loadCOCOExample(coco: COCO, image: COCO.Image, includeMasks: bool) = ObjectDetectionExample {
    let imgDir = coco.imagesDirectory
    let imgW = image["width"] :?> Int
    let imgH = image["height"] :?> Int
    let imgFileName = image["file_name"] :?> String
    let imgUrl: Uri? = nil
    if imgDir <> nil then
        let imgPath = imgDir! </> (imgFileName).path
        imgUrl = Uri(imgPath)!

    let imgId = image["id"] :?> Int
    let img = LazyImage(width: imgW, height: imgH, url: imgUrl)
    let annotations: [COCO.Annotation]
    if let anns = coco.imageToAnnotations[imgId] then
        annotations = anns
    else
        annotations = []

    let objects: [LabeledObject] = []
    objects.reserveCapacity(annotations.count)
    for annotation in annotations {
        let bb = annotation["bbox"] :?> [Double]
        let bbX = bb[0]
        let bbY = bb[1]
        let bbW = bb[2]
        let bbH = bb[3]
        let xMin = double(bbX) / double(imgW)
        let xMax = double(bbX + bbW) / double(imgW)
        let yMin = double(bbY) / double(imgH)
        let yMax = double(bbY + bbH) / double(imgH)
        let isCrowd: int?
        if let iscrowd = annotation["iscrowd"] then
            isCrowd = iscrowd as? Int
        else
            isCrowd = nil

        let area = double(annotation["area"] :?> Double)
        let classId = annotation["category_id"] :?> Int
        let classInfo = coco.categories[classId]!
        let className = classInfo["name"] :?> String
        let maskRLE: RLE?
        if includeMasks then
            maskRLE = coco.annotationToRLE(annotation)
        else
            maskRLE = nil

        let object = LabeledObject(
            xMin: xMin, xMax: xMax,
            yMin: yMin, yMax: yMax,
            className: className, classId: classId,
            isCrowd: isCrowd, area: area, maskRLE: maskRLE)
        objects.append(object)

    return ObjectDetectionExample(image: img, objects: objects)


fileprivate let makeBatch<BatchSamples: Collection>(
  samples: BatchSamples, device: Device,
  transform: (ObjectDetectionExample) = [ObjectDetectionExample]
) = [ObjectDetectionExample] where BatchSamples.Element = ObjectDetectionExample {
  return samples.reduce([]) = 
    $0 + transform($1)


*)
