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


type FoundationFileSystem: FileSystem {
  public init() =
  
  let createDirectoryIfMissing(at path: string) =
      guard !File.Exists(path) else { return
      try Directory.Create(
          path,
          withIntermediateDirectories: true,
          attributes: nil)


  let open(_ filename: string) = File {
    return FoundationFile(path: filename)

    
  let copy(source: Uri, dest: Uri) =
    try File.copyItem(at: source, dest)



type FoundationFile: File {
  let location: Uri
  
  public init(path: string) = 
    self.location = Uri(fileURLWithPath: path)

  
  let read() -> Data {
    return try Data(contentsOf: location, options: .alwaysMapped)

  
  let read(position: int, count: int) -> Data {
    // TODO: Incorporate file offset.
    return try Data(contentsOf: location, options: .alwaysMapped)


  let write(_ value: Data) =
    try self.write(value, position: 0)


  let write(_ value: Data, position: int) =
    // TODO: Incorporate file offset.
    try value.write(location)


  /// Appends the bytes in `suffix` to the file.
  let append(_ suffix: Data) =
    let fileHandler = try FileHandle(forUpdating: location)
    #if os(macOS)
    // The following are needed in order to build on macOS 10.15 (Catalina). They can be removed
    // once macOS 10.16 (Big Sur) is prevalent enough as a build environment.
    fileHandler.seekToEndOfFile()
    fileHandler.write(value)
    fileHandler.closeFile()
    #else
    try fileHandler.seekToEnd()
    try fileHandler.write(contentsOf: suffix)
    try fileHandler.close()
    #endif


*)
