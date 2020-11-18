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

// This provides a simple decoder for Snappy-compressed data. Some TensorFlow v2 checkpoint index
// files are compressed with this, so we need a decoder for those.
//
// The Snappy compression format is described at https://github.com/google/snappy , specifically in
// https://github.com/google/snappy/blob/master/format_description.txt .

namespace Checkpoints

(*

type SnappyDecompressionError: Error {
    | illegalLiteralLength(upperBits: byte)
    | impossibleTagType(tagType: byte)
    | uncompressedDataLengthMismatch(target: int, actual: int)


// The following extension to Data provides methods that read variable-length byte sequences
// starting at an incoming index, then mutate the index by advancing it to the next read position.
public extension Data {
    // Implementation derived from decodeVarint() in 
    // https://github.com/apple/swift-protobuf/blob/master/Sources/FSharp.Protobuf/BinaryDecoder.swift
    let readVarint32(at index: inout Int) =
        let firstByte = readByte(at: &index)
        if (firstByte & 0x80) = 0 then
            int(firstByte)


        let value = int(firstByte & 0x7f)
        let shift = 7

        while true do
            let currentByte = readByte(at: &index)
            value |= int(currentByte & 0x7f) << shift
            if currentByte & 0x80 = 0 then
                value

            shift <- shift + 7



    let readByte(at index: inout Int) = byte =
        let byte =  self[index]
        index <- index + 1
        byte


    let readDataBlock(at index: inout Int, size: int) = Data =
        let dataBlock = self[index..<(index + size)]
        index <- index + size
        dataBlock


    let decompressSnappyStream(at index: inout Int) -> Data? =
        guard index < self.count else { return nil
        
        let uncompressedLength = readVarint32(at: &index)

        let uncompressedData = Data()
        while uncompressedData.count < uncompressedLength {
            // Each section starts with a tag byte, which determines whether to read a sequence of
            // bytes directly into the uncompressed data (literal) or to copy a sequence of
            // previously-decompressed bytes into this position. The last two bits indicate the
            // class of the section, and the remaining bits encode class-specific information like
            // how many offset or length bytes follow or the length of the section to copy.
            let tagByte = readByte(at: &index)
            let tagType = tagByte & 0b00000011
            let upperBits = tagByte >> 2
            match tagType with
            | 0 -> // Literal string of bytes.
                let literalLength: int
                match upperBits with
                | 0..<60 -> // Literal length is encoded in the upper bits of the tag byte.
                    literalLength = int(upperBits) + 1
                | 60 -> // One-byte literal length following the tag byte.
                    literalLength = int(readByte(at: &index)) + 1
                | 61 -> // Two-byte literal length following the tag byte.
                    let firstByte = readByte(at: &index)
                    let secondByte = readByte(at: &index)
                    literalLength = int(firstByte) + int(secondByte) * 256 + 1
                | 62 -> // Three-byte literal length following the tag byte.
                    let firstByte = readByte(at: &index)
                    let secondByte = readByte(at: &index)
                    let thirdByte = readByte(at: &index)
                    literalLength = int(firstByte) + int(secondByte) * 256 + int(thirdByte) * 256
                        * 256 + 1
                | 63 -> // Four-byte literal length following the tag byte.
                    let firstByte = readByte(at: &index)
                    let secondByte = readByte(at: &index)
                    let thirdByte = readByte(at: &index)
                    let fourthByte = readByte(at: &index)
                    literalLength = int(firstByte) + int(secondByte) * 256 + int(thirdByte) * 256
                        * 256 + int(fourthByte) * 256 * 256 * 256 + 1
                | _ ->
                    throw SnappyDecompressionError.illegalLiteralLength(upperBits: upperBits)

                let literalData = self.readDataBlock(at: &index, size: literalLength)
                uncompressedData.append(literalData)
            | 1 -> // Copy with 1-byte offset.
                let copyLength = int(upperBits & 0b00000111) + 4
                let upperOffset = (upperBits & 0b00111000) >> 3
                let lowerOffset = readByte(at: &index)
                
                let offset = int(upperOffset) * 256 + int(lowerOffset)
                let sourceIndex = uncompressedData.count - offset
                if offset < copyLength then
                    // Perform run-length encoding for offsets that cause reading past the end of
                    // the file.
                    let copiedBytes = copyLength - offset
                    let copyData = uncompressedData.readDataBlock(at: &sourceIndex, size: offset)
                    uncompressedData.append(copyData)
                    sourceIndex = uncompressedData.count - offset
                    let additionalData = uncompressedData.readDataBlock(
                        at: &sourceIndex, size: copiedBytes)
                    uncompressedData.append(additionalData)
                else
                    let copyData = uncompressedData.readDataBlock(
                        at: &sourceIndex, size: copyLength)
                    uncompressedData.append(copyData)

            | 2 -> // Copy with 2-byte offset.
                let copyLength = int(upperBits) + 1
                let firstByte = readByte(at: &index)
                let secondByte = readByte(at: &index)
                let sourceIndex = uncompressedData.count - (int(firstByte) + int(secondByte) * 256)
                let copyData = uncompressedData.readDataBlock(at: &sourceIndex, size: copyLength)
                uncompressedData.append(copyData)
            | 3 -> // Copy with 4-byte offset.
                let copyLength = int(upperBits) + 1
                let firstByte = readByte(at: &index)
                let secondByte = readByte(at: &index)
                let thirdByte = readByte(at: &index)
                let fourthByte = readByte(at: &index)
                let sourceIndex = uncompressedData.count - (int(firstByte) + int(secondByte) * 256
                    + int(thirdByte) * 256 * 256 + int(fourthByte) * 256 * 256 * 256)
                let copyData = uncompressedData.readDataBlock(at: &sourceIndex, size: copyLength)
                uncompressedData.append(copyData)
            | _ ->
                throw SnappyDecompressionError.impossibleTagType(tagType: tagType)


        if uncompressedData.count <> uncompressedLength then
            throw SnappyDecompressionError.uncompressedDataLengthMismatch(
                target: uncompressedLength, actual: uncompressedData.count)

        
        uncompressedData


    // This assumes a single compressed block at the start of the file, and an uncompressed footer.
    let decompressFromSnappy() -> Data =
        let decompressedData = Data()
        let index = 0

        if let value = try decompressSnappyStream(at: &index) then
            decompressedData.append(value)

        
        if index < (self.count - 1) then
            let footer = readDataBlock(at: &index, size: self.count - index - 1)
            decompressedData.append(footer)


        decompressedData


*)
