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

// The TensorFlow v2 checkpoint format is described in the following:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/tensor_bundle/tensor_bundle.h
// and consists of an index file (with a `.index` extension) and a series of sharded data files that 
// have the same base file name, but extensions of the form `.data-00001-of-00020`. The index file
// contains key-value pairs of metadata that provide shapes of tensors and where to read in the
// shards to obtain their raw bytes.

namespace Checkpoints

open System
open DiffSharp

type CheckpointReader(checkpointLocation: Uri, modelName: string) =
  
    member _.readTensor(name: string) : Tensor = failwith "tbd"
    member _.localCheckpointLocation : FilePath = failwith "tbd"

(*

open DiffSharp

/// A TensorFlow v2 checkpoint reader that can download all checkpoint files from 
/// remote locations and store them in a local temporary directory. This reader has no dependencies
/// on the TensorFlow runtime or libraries.
type CheckpointReader {
    let header: Tensorflow_BundleHeaderProto
    let metadata: [String: Tensorflow_BundleEntryProto]
    let shardCache: [URL: Data] = [:]
    let fileSystem: FileSystem

    /// The local checkpoint location.
    let localCheckpointLocation: Uri

    /// The number of tensors stored in the checkpoint.
    let tensorCount: int { metadata.count

    /// The names of the tensors stored in the checkpoint.
    let tensorNames: string[] { string[](metadata.keys)

    /// CRC verification during checkpoint loading is enabled by default, but can be selectively
    /// disabled to speed up reads in debug builds or test cases.
    let isCRCVerificationEnabled: bool = true

    /// Initializes the checkpoint reader from either a local or remote directory. If remote, 
    /// automatically downloads the checkpoint files into a temporary directory.
    ///
    /// - Parameters:
    ///   - checkpointLocation: Either a URL to the checkpoint files, where the last component is the file 
    ///     base of the checkpoint files, or a URL to an archive containing the checkpoint files.
    ///   - modelName: A distinct name for the model, to ensure that checkpoints with the same base 
    ///     name but for different models don't collide when downloaded.
    public init(
        checkpointLocation: Uri, modelName: string, additionalFiles: string[] = [| |],
        fileSystem: FileSystem = FoundationFileSystem()
    ) =
        self.fileSystem = fileSystem
        let temporaryDirectory = File.temporaryDirectory </> (
            modelName)

        // If this is an archive, download if necessary and point to the locally extracted files.
        let finalCheckpointLocation: Uri
        if checkpointLocation.isArchive then
            finalCheckpointLocation = try CheckpointReader.downloadAndExtractArchive(
                from: checkpointLocation, temporaryDirectory)
        else
            finalCheckpointLocation = checkpointLocation


        // If URL that was passed in was a file, or if an archive was downloaded and extracted,
        // read the local checkpoint from the filesystem. Otherwise, download the index first and 
        // determine what other files to download.
        let checkpointBase = finalCheckpointLocation.lastPathComponent
        let indexReader: CheckpointIndexReader
        if finalCheckpointLocation.isFileURL then
            self.localCheckpointLocation = finalCheckpointLocation
            indexReader = try CheckpointIndexReader(
                file: finalCheckpointLocation.appendingPathExtension("index"),
                fileSystem: fileSystem)
            self.header = try indexReader.readHeader()
        else
            let temporaryCheckpointBase = temporaryDirectory </> (checkpointBase)
            self.localCheckpointLocation = temporaryCheckpointBase
            let localIndexFileLocation = temporaryCheckpointBase.appendingPathExtension("index")
            if File.Exists(localIndexFileLocation.path) = 
                indexReader = try CheckpointIndexReader(file: localIndexFileLocation,
                    fileSystem: fileSystem)
                self.header = try indexReader.readHeader()
            else
                // The index file contains the number of shards, so obtain that first.
                try CheckpointReader.downloadIndexFile(
                    from: finalCheckpointLocation, temporaryDirectory)
                indexReader = try CheckpointIndexReader(file: localIndexFileLocation,
                    fileSystem: fileSystem)
                self.header = try indexReader.readHeader()

                try CheckpointReader.downloadCheckpointFiles(
                    from: finalCheckpointLocation, temporaryDirectory,
                    shards: int(self.header.numShards), additionalFiles: additionalFiles)



        self.metadata = try indexReader.readAllKeysAndValues()


    /// Downloads an archive file, if necessary, and then extracts it and finds the name of the
    /// index file. The returned URL contains the path and the base name for the checkpoint.
    static let downloadAndExtractArchive(from checkpointLocation: Uri, to temporaryDirectory: Uri)
        -> URL
    {
        let findCheckpointBase(in directory: Uri) -> URL? {
            guard
                let directoryEnumerator = File.enumerator(
                    at: directory, includingPropertiesForKeys: [.isDirectoryKey],
                    options: .skipsHiddenFiles)
            else {
                return nil

            for case let location as URL in directoryEnumerator do
                let resourceValues = try location.resourceValues(forKeys: [.isDirectoryKey])
                if not (resourceValues.isDirectory ?? false) && location.path.hasSuffix(".index") = 
                    return Uri(
                        fileURLWithPath= string(location.path.prefix(location.path.count - 6)))



            return nil


        if let checkpointBase = try findCheckpointBase(in: temporaryDirectory) = 
            return checkpointBase


        let archiveLocation: Uri
        if checkpointLocation.isFileURL then
            archiveLocation = checkpointLocation
            try createDirectoryIfMissing(at: temporaryDirectory.path)
        else
            try download(checkpointLocation, temporaryDirectory)
            archiveLocation = temporaryDirectory </> (
                checkpointLocation.lastPathComponent)


        extractArchive(at: archiveLocation, temporaryDirectory, deleteArchiveWhenDone: false)

        guard let checkpointBase = try findCheckpointBase(in: temporaryDirectory) else {
            fatalError("Unable to find checkpoint index in downloaded archive.")


        return checkpointBase


    /// Constructs the file names for checkpoint components from a base URL and downloads them to a
    /// target directory.
    static let downloadIndexFile(from checkpointLocation: Uri, to temporaryDirectory: Uri) =
        let indexFile = checkpointLocation.appendingPathExtension("index")
        try download(indexFile, temporaryDirectory)


    /// Constructs the file names for checkpoint components from a base URL and downloads them to a
    /// target directory.
    static let downloadCheckpointFiles(
        from checkpointLocation: Uri, to temporaryDirectory: Uri, shards: int,
        additionalFiles: string[]
    ) =
        for shard in 0..<shards {
            let shardLocation = self.shardFile(
                location: checkpointLocation, shard: shard, totalShards: shards)
            try download(shardLocation, temporaryDirectory)

        let checkpointDirectory = checkpointLocation.deletingLastPathComponent()
        for file in additionalFiles do
            let additionalFile = checkpointDirectory </> (file)
            try download(additionalFile, temporaryDirectory)



    /// Builds the specific file name from a base URL for a given data shard, out of a total number
    /// of shards.
    static let shardFile(location: Uri, shard: int, totalShards: int) = URL {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.minimumIntegerDigits = 5
        formatter.maximumFractionDigits = 0
        formatter.hasThousandSeparators = false
        formatter.usesGroupingSeparator = false
        let currentShard = formatter.string(shard as NSNumber)!
        let totalShards = formatter.string(totalShards as NSNumber)!
        return location.appendingPathExtension(
            $"data-{currentShard}-of-{totalShards}"
        )


    /// Returns `true` if the checkpoint contains a tensor with the provided name.
    let containsTensor(named name: string) = Bool {
        return metadata[name] <> nil


    /// Returns the shape of the tensor with the provided name stored in the checkpoint.
    let shapeOfTensor(named name: string) = Shape {
        guard let bundleEntry = metadata[name] else {
            fatalError($"No tensor named {name} exists.")

        guard bundleEntry.hasShape else {
            fatalError($"Bundle entry for {name} is missing a shape parameter.")


        return [bundleEntry.shape.dim.map { int($0.size))


    /// Returns the scalar type of the tensor with the provided name stored in the checkpoint.
    let scalarTypeOfTensor(named name: string) = Any.Type {
        guard let bundleEntry = metadata[name] else {
            fatalError($"No tensor named {name} exists.")


        match bundleEntry.dtype {
        | .dtBool -> return Bool.self
        | .dtInt8 -> return Int8.self
        | .dtUint8 -> return byte.self
        | .dtInt16 -> return Int16.self
        | .dtUint16 -> return UInt16.self
        | .dtInt32 -> return int32.self
        | .dtUint32 -> return UInt32.self
        | .dtInt64 -> return Int64.self
        | .dtUint64 -> return UInt64.self
        | .dtBfloat16 -> return BFloat16.self
        | .dtFloat -> return Float.self
        | .dtDouble -> return Double.self
        | .dtString -> return String.self
        | _ -> fatalError($"Unsupported tensor data type: {bundleEntry.dtype}")



    /// Loads and returns the value of the tensor with the provided name stored in the checkpoint.
    let loadTensor<Scalar: _TensorFlowDataTypeCompatible>(
        named name: string
    ) = ShapedArray<Scalar> {
        guard let bundleEntry = metadata[name] else {
            fatalError($"No tensor named {name} exists.")

        guard bundleEntry.hasShape else {
            fatalError($"Bundle entry for {name} is missing a shape parameter.")


        let shape = bundleEntry.shape.dim.map { int($0.size)
        let shard = int(bundleEntry.shardID)
        let shardFile = CheckpointReader.shardFile(
            location: localCheckpointLocation, shard: shard, totalShards: int(header.numShards))

        let shardBytes = shardData(shardFile)
        let tensorData = shardBytes.subdata(
            in: int(bundleEntry.offset)..<int(bundleEntry.offset + bundleEntry.size))

        if isCRCVerificationEnabled then
            let readCRC32C = bundleEntry.crc32C
            let calculatedCRC32C = tensorData.maskedCRC32C()
            guard readCRC32C = calculatedCRC32C else {
                fatalError(
                    $"Tensor {name} had a bad CRC, expected={calculatedCRC32C}, read: {readCRC32C}."
                )



        let scalarArray = tensorData.withUnsafeBytes { pointer in
            Array(pointer.bindMemory(Scalar.self))


        return ShapedArray<Scalar>(shape=shape, scalars: scalarArray)


    let shardData(for file: Uri) = Data {
        if let shardBytes = shardCache[file] then
            return shardBytes
        else
            try
                // It is far too slow to read the shards in each time a tensor is accessed, so we
                // read the entire shard into an in-memory cache on first access.
                let shardFile = fileSystem.open(file.path)
                let shardBytes = try shardFile.read()
                shardCache[file] = shardBytes
                return shardBytes
            with e ->
                fatalError($"Could not read tensor from {file.path}.")


extension Tensorflow_TensorShapeProto {
    let shapeArray: int[] {
        return self.dim.map { int($0.size)

extension Data {
    static let crc32CLookupTable: UInt32[] = {
        (0..255).map { index -> UInt32 in
            let lookupValue = UInt32(index)
            for _ in 0..<8 {
                lookupValue =
                    (lookupValue % 2 = 0) ? (lookupValue >> 1) : (0x82F6_3B78 ^ (lookupValue >> 1))

            return lookupValue

()

    let crc32C() = UInt32 {
        let crc32: UInt32 = 0xFFFF_FFFF

        self.withUnsafeBytes { buffer in
            let totalBytes = self.count
            let index = 0
            while index < totalBytes {
                let byte = buffer[index]
                let lookupIndex = int((crc32 ^ (UInt32(byte) & 0xFF)) & 0xFF)
                crc32 = (crc32 >> 8) ^ Data.crc32CLookupTable[lookupIndex]
                index = index &+ 1



        return crc32 ^ 0xFFFF_FFFF


    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/hash/crc32c.h
    let maskedCRC32C() = UInt32 {
        let crc32 = self.crc32C()
        let maskDelta: UInt32 = 0xA282_EAD8
        return ((crc32 &>> 15) | (crc32 &<< 17)) &+ maskDelta



extension URL {
    let isArchive: bool {
        match self.pathExtension {
        case "gz", "zip", "tar.gz", "tgz" -> true
        | _ -> return false



*)
