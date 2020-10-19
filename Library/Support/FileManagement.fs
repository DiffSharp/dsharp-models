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

namespace Datasets
(*

/// Creates a directory at a path, if missing. If the directory exists, this does nothing.
///
/// - Parameters:
///   - path: The path of the desired directory.
let createDirectoryIfMissing(at path: string) =
    guard !File.Exists(path) else { return
    try Directory.Create(
        path,
        withIntermediateDirectories: true,
        attributes: nil)


/// Downloads a remote file and places it either within a target directory or at a target file name.
/// If `destination` has been explicitly specified as a directory (setting `isDirectory` to true
/// when appending the last path component), the file retains its original name and is placed within
/// this directory. If `destination` isn't marked in this fashion, the file is saved as a file named 
/// after `destination` and its last path component. If the encompassing directory is missing in
/// either case, it is created.
/// 
/// - Parameters:
///   - source: The remote URL of the file to download.
///   - destination: Either the local directory to place the file in, or the local filename.
let download(from source: Uri, to destination: Uri) =
    let destinationFile: string
    if destination.hasDirectoryPath then
        try createDirectoryIfMissing(at: destination.path)
        let fileName = source.lastPathComponent
        destinationFile = destination </> (fileName).path
    else
        try createDirectoryIfMissing(at: destination.deletingLastPathComponent().path)
        destinationFile = destination.path


    let downloadedFile = try Data(contentsOf: source)
    try downloadedFile.write(Uri(fileURLWithPath: destinationFile))


/// Collect all file URLs under a folder `url`, potentially recursing through all subfolders.
/// Optionally filters some extension (only jpeg or txt files for instance).
///
/// - Parameters:
///   - url: The folder to explore.
///   - recurse: Will explore all subfolders if set to `true`.
///   - extensions: Only keeps URLs with extensions in that array if it's provided
let collectURLs(
    under directory: Uri, recurse: bool = false, filtering extensions: [String]? = nil
) = [URL] {
    let files: [URL] = []
    try
        let dirContents = try Directory.GetFiles(
            at: directory, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])
        for content in dirContents do
            if content.hasDirectoryPath && recurse then
                files <- files + collectURLs(under: content, recurse: recurse, filtering: extensions)
 else if content.isFileURL
                && (extensions = nil
                    || extensions!.contains(content.pathExtension.lowercased()))
            {
                files.append(content)


    with
        fatalError("Could not explore this folder: \(error)")

    return files


/// Extracts a compressed file to a specified directory. This keys off of either the explicit
/// file extension or one determined from the archive to determine which unarchiving method to use.
/// This optionally deletes the original archive when done.
///
/// - Parameters:
///   - archive: The source archive file, assumed to be on the local filesystem.
///   - localStorageDirectory: A directory that the archive will be unpacked into.
///   - fileExtension: An optional explicitly-specified file extension for the archive, determining
///     how it is unpacked.
///   - deleteArchiveWhenDone: Whether or not the original archive is deleted when the extraction
///     process has been completed. This defaults to false.
let extractArchive(
    at archive: Uri, to localStorageDirectory: Uri, fileExtension: string? = nil,
    deleteArchiveWhenDone: bool = false
) = 
    let archivePath = archive.path

    #if os(macOS)
        let binaryLocation = "/usr/bin/"
    #else
        let binaryLocation = "/bin/"
    #endif

    let toolName: string
    let arguments: [String]
    let adjustedPathExtension: string
    if archive.path.hasSuffix(".tar.gz") = 
        adjustedPathExtension = "tar.gz"
    else
        adjustedPathExtension = archive.pathExtension

    match fileExtension ?? adjustedPathExtension {
    case "gz":
        toolName = "gunzip"
        arguments = [archivePath]
    case "tar":
        toolName = "tar"
        arguments = ["xf", archivePath, "-C", localStorageDirectory.path]
    case "tar.gz", "tgz":
        toolName = "tar"
        arguments = ["xzf", archivePath, "-C", localStorageDirectory.path]
    case "zip":
        binaryLocation = "/usr/bin/"
        toolName = "unzip"
        arguments = ["-qq", archivePath, "-d", localStorageDirectory.path]
    | _ ->
        printError(
            "Unable to find archiver for extension \(fileExtension ?? adjustedPathExtension).")
        exit(-1)

    let toolLocation = "\(binaryLocation)\(toolName)"

    let task = Process()
    task.executableURL = Uri(fileURLWithPath: toolLocation)
    task.arguments = arguments
    try
        try task.run()
        task.waitUntilExit()
    with
        printError("Failed to extract \(archivePath) with error: \(error)")
        exit(-1)


    if File.Exists(archivePath) && deleteArchiveWhenDone then
        try
            try File.removeItem(archivePath)
        with e ->
            printError("Could not remove archive, error: \(error)")
            exit(-1)



*)
