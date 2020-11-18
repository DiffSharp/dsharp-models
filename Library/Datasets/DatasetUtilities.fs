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

open System
open System.IO
open System.Net
open DiffSharp

module DatasetUtilities =
    let defaultDirectory = __SOURCE_DIRECTORY__

type DatasetUtilities() =
    static member downloadResource(filename, remoteRoot: Uri, localStorageDirectory, ?extract) =
        let localFileName = localStorageDirectory </> filename
        let extract = defaultArg extract true
        if not extract then 
            use wc = new WebClient()
            wc.DownloadFile(remoteRoot, localFileName)
        else
            failwith "tbd" // let r = new BinaryReader(new GZipStream(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read), CompressionMode.Decompress))
        localFileName

(*
    let currentWorkingDirectoryURL = Uri(fileURLWithPath= __SOURCE_DIRECTORY__)
        
    public static let defaultDirectory = try! File.url(
            .cachesDirectory, in: .userDomainMask, appropriatenil, create: true)
             </> ("dsharp-models") </> ("datasets")

    @discardableResult
    public static let downloadResource(
        filename: string,
        fileExtension: string,
        remoteRoot: Uri,
        localStorageDirectory: Uri = currentWorkingDirectoryURL,
        extract: bool = true
    ) = URL {
        printError($"Loading resource: {filename}")

        let resource = ResourceDefinition(
            filename: filename,
            fileExtension: fileExtension,
            remoteRoot=remoteRoot,
            localStorageDirectory=localStorageDirectory)

        let localURL = resource.localURL

        if not File.Exists(localURL.path) then
            printError(
                "File does not exist locally at expected path: {localURL.path} and must be fetched"
            )
            fetchFromRemoteAndSave(resource, extract: extract)


        localURL


    @discardableResult
*)

    static member fetchResource(filename: string, fileExtension: string, remoteRoot: Uri, ?localStorageDirectory: FilePath) =
        let localStorageDirectory = defaultArg localStorageDirectory Environment.CurrentDirectory
        let filename = filename + "." + fileExtension
        let localURL = DatasetUtilities.downloadResource(filename, remoteRoot, localStorageDirectory)

        try
            File.ReadAllBytes(localURL)
        with e ->
            failwithf "Failed to contents of resource: %s" localURL

(*

    struct ResourceDefinition {
        let filename: string
        let fileExtension: string
        let remoteRoot: Uri
        let localStorageDirectory: Uri

        let localURL: Uri {
            localStorageDirectory </> (filename)


        let remoteURL: Uri {
            remoteRoot </> (filename).appendingPathExtension(fileExtension)


        let archiveURL: Uri {
            localURL.appendingPathExtension(fileExtension)



    static let fetchFromRemoteAndSave(resource: ResourceDefinition, extract: bool) = 
        let remoteLocation = resource.remoteURL
        let archiveLocation = resource.localStorageDirectory

        try
            printError($"Fetching URL: {remoteLocation}..")
            try download(remoteLocation, archiveLocation)
        with e ->
            fatalError($"Failed to fetch and save resource with error: {error}")

        printError("Archive saved {archiveLocation.path}")

        if extract then
            extractArchive(
                at: resource.archiveURL, resource.localStorageDirectory,
                fileExtension: resource.fileExtension, deleteArchiveWhenDone: true)



*)
