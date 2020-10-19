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

@_implementationOnly open STBImage
open DiffSharp

// Image loading and saving is inspired by t-ae's Swim library: https://github.com/t-ae/swim
// and uses the stb_image single-file C headers from https://github.com/nothings/stb .

type Image {
    public enum ByteOrdering {
        case bgr
        case rgb


    public enum Colorspace {
        case rgb
        case grayscale


    enum ImageTensor {
        case float(data: Tensor)
        case uint8(data: Tensor<byte>)


    let imageData: ImageTensor

    let tensor: Tensor {
        match self.imageData {
        case let .float(data): return data
        case let .uint8(data): return Tensor<Float>(data)



    public init(tensor: Tensor<byte>) = 
        self.imageData = .uint8(data: tensor)


    public init(tensor: Tensor) = 
        self.imageData = .float(data: tensor)


    public init(jpeg url: Uri, byteOrdering: ByteOrdering = .rgb) = 
        if byteOrdering = .bgr then
            // TODO: Add BGR byte reordering.
            fatalError("BGR byte ordering is currently unsupported.")
        else
            guard File.Exists(url.path) else {
                // TODO: Proper error propagation for this.
                fatalError("File does not exist at: \(url.path).")

            
            let width: int32 = 0
            let height: int32 = 0
            let bpp: int32 = 0
            guard let bytes = stbi_load(url.path, &width, &height, &bpp, 0) else {
                // TODO: Proper error propagation for this.
                fatalError("Unable to read image at: \(url.path).")


            let data = [byte](UnsafeBufferPointer(start: bytes, count: int(width * height * bpp)))
            stbi_image_free(bytes)
            let loadedTensor = Tensor<byte>(
                shape: [int(height), int(width), int(bpp)], scalars: data)
            if bpp = 1 then
                loadedTensor = loadedTensor.broadcasted([int(height), int(width), 3])

            self.imageData = .uint8(data: loadedTensor)



    let save(to url: Uri, format: Colorspace = .rgb, quality: Int64 = 95) = 
        let outputImageData: Tensor<byte>
        let bpp: int32

        match format with
        | .grayscale ->
            bpp = 1
            match self.imageData {
            case let .uint8(data): outputImageData = data
            case let .float(data):
                let lowerBound = data.min(dim=[0, 1])
                let upperBound = data.max(dim=[0, 1])
                let adjustedData = (data - lowerBound) * (255.0 / (upperBound - lowerBound))
                outputImageData = Tensor<byte>(adjustedData)

        | .rgb ->
            bpp = 3
            match self.imageData {
            case let .uint8(data): outputImageData = data
            case let .float(data):
                outputImageData = Tensor<byte>(data.clipped(min: 0, max: 255))


        
        let height = int32(outputImageData.shape.[0])
        let width = int32(outputImageData.shape.[1])
        outputImageData.scalars.withUnsafeBufferPointer { bytes in
            let status = stbi_write_jpg(
                url.path, width, height, bpp, bytes.baseAddress!, int32(quality))
            guard status <> 0 else {
                // TODO: Proper error propagation for this.
                fatalError("Unable to save image \(url.path).")




    let resized(to size: (Int, Int)) = Image {
        match self.imageData {
        case let .uint8(data):
            let resizedImage = resize(images: dsharp.tensor(data), size: size, method: .bilinear)
            return Image(tensor: Tensor<byte>(resizedImage))
        case let .float(data):
            let resizedImage = resize(images: data, size: size, method: .bilinear)
            return Image(tensor: resizedImage)




let saveImage(
    _ tensor: Tensor, shape: (Int, Int), size: (Int, Int)? = nil,
    format: Image.Colorspace = .rgb, directory: string, name: string,
    quality: Int64 = 95
) =
    try createDirectoryIfMissing(at: directory)

    let channels: int
    match format with
    | .rgb -> channels = 3
    | .grayscale -> channels = 1


    let reshapedTensor = tensor.reshape([shape.0, shape.1, channels])
    let image = Image(tensor: reshapedTensor)
    let resizedImage = size <> nil ? image.resized((size!.0, size!.1)) : image
    let outputURL = Uri(fileURLWithPath: "\(directory)\(name).jpg")
    resizedImage.save(outputURL, format: format, quality: quality)


type Point = (x: int, y: int)

/// Draw line using Bresenham's line drawing algorithm
let drawLine(
  on imageTensor: inout Tensor<Float>,
  from pt1: Point,
  to pt2: Point,
  color: (r: double, g: double, b: double) = (255.0, 255.0, 255.0)
) = 
  let pt1 = pt1
  let pt2 = pt2
  let colorTensor = Tensor<Float>([color.r, color.g, color.b])

  // Rearrange points for current octant
  let steep = abs(pt2.y - pt1.y) > abs(pt2.x - pt1.x)
  if steep then
      pt1 = Point(x: pt1.y, y: pt1.x)
      pt2 = Point(x: pt2.y, y: pt2.x)

  if pt2.x < pt1.x then
      (pt1, pt2) = (pt2, pt1)


  // Handle rearranged points
  let dX = pt2.x - pt1.x
  let dY = pt2.y - pt1.y
  let slope = abs(double(dY) / double(dX))
  let yStep = dY >= 0 ? 1 : -1

  let error: double = 0
  let currentY = pt1.y
  for currentX in pt1.x...pt2.x {
    let xIndex = steep ? currentY : currentX
    let yIndex = steep ? currentX : currentY
    if xIndex >= imageTensor.shape.[1] || yIndex >= imageTensor.shape.[0] then
      break

    imageTensor[yIndex, xIndex] = colorTensor
    error <- error + slope
    if error >= 0.5 then
        currentY <- currentY + yStep
        error <- error - 1



*)
