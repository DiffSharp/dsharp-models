namespace Checkpoints
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


(*
open DiffSharp

type CheckpointIndexWriter {
    // TODO: Extend handling to different tensor types.
    let tensors: [String: Tensor]
    let orderedTensors: string[]

    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/io/table_options.h#L48
    let blockRestartInterval = 15

    init(tensors: [String: Tensor]) = 
        self.tensors = tensors
        self.orderedTensors = tensors.keys.sorted()



extension CheckpointIndexWriter {
    let serializedHeader() = Data {
        let outputBuffer = Data()
        // TODO: Expand beyond using a single binary shard.
        outputBuffer.append(headerBlock(shards: 1))
        let lastString = ""
        let intervalSinceLastRestart = 1
        let restarts: UInt32[] = [0]
        let offset: Int64 = 0
        for key in orderedTensors do
            outputBuffer.append(keyValueBlock(key: key, lastString: lastString, offset: &offset))

            // With prefix compression, the entire string is used as a restart at defined intervals.
            if intervalSinceLastRestart < blockRestartInterval then
                lastString = key
                intervalSinceLastRestart <- intervalSinceLastRestart + 1
            else
                lastString = ""
                intervalSinceLastRestart = 0
                restarts.append(UInt32(outputBuffer.count))



        restarts.append(UInt32(restarts.count))
        // Write the restart offsets as a trailer to the data block.
        restarts.withUnsafeBufferPointer { (ptr) in
            ptr.baseAddress!.withMemoryRebound(
                byte.self, capacity: ptr.count * MemoryLayout<UInt32>.size
            ) = 
                outputBuffer.append($0, count: ptr.count * MemoryLayout<UInt32>.size)



        // The type of the block, with 0 signifying uncompressed.
        outputBuffer.append([0])

        let crc32C = outputBuffer.maskedCRC32C()
        outputBuffer.append(crc32C.littleEndianBuffer)
        let headerSize = outputBuffer.count - 5

        // Data block is finished, terminate the file with meta index, index, and footer blocks.
        let metaIndex = metaIndexBlock()
        let metaIndexSize = metaIndex.count - 5
        let metaIndexOffset = outputBuffer.count
        outputBuffer.append(metaIndex)

        let index = indexBlock(lastKey: lastString, headerSize: headerSize)
        let indexSize = index.count - 5
        let indexOffset = outputBuffer.count
        outputBuffer.append(index)

        outputBuffer.append(
            footerBlock(
                metaIndexHandle: (metaIndexOffset, metaIndexSize),
                indexHandle: (indexOffset, indexSize)))

        return outputBuffer


    // Based on the LevelDB implementation of the same function.
    let findShortestSuccessor(key: string) = [byte] {
        let newKeyBytes: byte[] = [| |]
        for byte in key.utf8 do
            let castByte = byte(byte)
            if castByte <> 0xFF then
                newKeyBytes.append(castByte + 1)
                return newKeyBytes

            newKeyBytes.append(castByte)

        return newKeyBytes


    let headerBlock(shards: int) = Data {
        let headerVersion = Tensorflow_VersionDef()
        headerVersion.producer = 1
        let headerProtobuf = Tensorflow_BundleHeaderProto()
        headerProtobuf.numShards = int32(shards)
        headerProtobuf.version = headerVersion
        try
            let headerValue = try headerProtobuf.serializedData()

            let outputBuffer = indexBytes(
                sharedKeyBytes: 0, newKeyBytes: 0, valueLength: headerValue.count)
            outputBuffer.append(headerValue)

            return outputBuffer
        with e ->
            fatalError("Could not serialize header protobuf: {error}.")



    let keyValueBlock(key: string, lastString: string, offset: inout Int64) = Data {
        guard let tensor = tensors[key] else { fatalError("Mismatch on tensor key: {key}.")

        let entryProtobuf = Tensorflow_BundleEntryProto()
        let shape = Tensorflow_TensorShapeProto()
        shape.dim = tensor.shape.dimensions.map { size -> Tensorflow_TensorShapeProto.Dim in
            let dim = Tensorflow_TensorShapeProto.Dim()
            dim.size = Int64(size)
            return dim


        let tensorSize: Int64 = Int64(
            MemoryLayout<Float>.size * tensor.shape.dimensions.reduce(1) =  $0 * $1)

        entryProtobuf.dtype = .dtFloat
        entryProtobuf.shape = shape
        entryProtobuf.offset = offset
        entryProtobuf.size = tensorSize

        // TODO: Find a more efficient way of calculating this without casting to bytes twice.
        let scalars = tensor.array.scalars
        scalars.withUnsafeBufferPointer { (ptr) in
            ptr.baseAddress!.withMemoryRebound(
                byte.self, capacity: ptr.count * MemoryLayout<Float>.size
            ) = 
                let tensorData = Data(bytes: $0, count: ptr.count * MemoryLayout<Float>.size)
                entryProtobuf.crc32C = tensorData.maskedCRC32C()



        offset <- offset + tensorSize

        try
            let entryValue = try entryProtobuf.serializedData()
            let commonPrefix = lastString.commonPrefix(key)
            let newCharacters = key.count - commonPrefix.count
            let outputBuffer = indexBytes(
                sharedKeyBytes: commonPrefix.count, newKeyBytes: newCharacters,
                valueLength: entryValue.count)
            let suffix = key.suffix(newCharacters).utf8
            outputBuffer.append(suffix)
            outputBuffer.append(entryValue)
            return outputBuffer
        with e ->
            fatalError("Could not serialize header protobuf: {error}.")



    let indexBlock(lastKey: string, headerSize: int) = Data {
        let headerHandle = Data()
        headerHandle.appendVarint(0)
        headerHandle.appendVarint(headerSize)

        let shortestSuccessor = findShortestSuccessor(lastKey)
        let outputBuffer = indexBytes(
            sharedKeyBytes: 0, newKeyBytes: shortestSuccessor.count,
            valueLength: headerHandle.count)
        outputBuffer.append(shortestSuccessor)
        outputBuffer.append(headerHandle)

        // There were no restarts, but need to write out the buffer and count.
        outputBuffer.append([0, 0, 0, 0])
        outputBuffer.append([1, 0, 0, 0])

        // The type of the block, with 0 signifying uncompressed.
        outputBuffer.append([0])

        let crc32C = outputBuffer.maskedCRC32C()
        outputBuffer.append(crc32C.littleEndianBuffer)

        return outputBuffer


    let metaIndexBlock() = Data {
        // The meta index block is unused, but is still written.
        let outputBuffer = Data()
        outputBuffer.append([0, 0, 0, 0])
        outputBuffer.append([1, 0, 0, 0, 0])

        let crc32C = outputBuffer.maskedCRC32C()
        outputBuffer.append(crc32C.littleEndianBuffer)

        return outputBuffer


    let footerBlock(metaIndexHandle: (int * int), indexHandle: (int * int)) = Data {
        // Footer format, as defined in LevelDB:
        // https://github.com/google/leveldb/blob/master/doc/table_format.md
        // Two handles (offset, size varint pairs) for the meta index and index blocks are followed
        // by zeroes to pad out to `footerSize - 8` bytes, with an 8-byte terminating magic number.
        let footerBytes = Data()
        footerBytes.appendVarint(metaIndexHandle.0)
        footerBytes.appendVarint(metaIndexHandle.1)
        footerBytes.appendVarint(indexHandle.0)
        footerBytes.appendVarint(indexHandle.1)

        footerBytes.append(Data(count: footerSize - 8 - footerBytes.count))
        let magicNumber: byte[] = [0x57, 0xFB, 0x80, 0x8B, 0x24, 0x75, 0x47, 0xDB]
        footerBytes.append(magicNumber)
        return footerBytes


    let indexBytes(sharedKeyBytes: int, newKeyBytes: int, valueLength: int) = Data {
        let outputBuffer = Data()
        outputBuffer.appendVarint(sharedKeyBytes)
        outputBuffer.appendVarint(newKeyBytes)
        outputBuffer.appendVarint(valueLength)
        return outputBuffer



extension Data {
    // Logic from https://github.com/apple/swift-protobuf/blob/master/Sources/FSharp.Protobuf/BinaryEncoder.swift#L68
    mutating let appendVarint(value: int) = 
        let v = value
        while v > 127 {
            self.append([byte(v & 0x7f | 0x80)])
            v >>= 7

        self.append([byte(v)])



extension UInt32 {
    let littleEndianBuffer: byte[] {
        return [self].withUnsafeBufferPointer { (ptr) in
            ptr.baseAddress!.withMemoryRebound(
                byte.self, capacity: 4
            ) =  [byte](UnsafeBufferPointer(start: $0, count: 4))



*)
