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


type IFileSystem {
  /// Creates a directory at a path, if missing. If the directory exists, this does nothing.
  ///
  /// - Parameters:
  ///   - path: The path of the desired directory.
  let createDirectoryIfMissing(at path: string)

  /// Opens a file at the specified location for reading or writing.
  ///
  /// - Parameters:
  ///   - path: The path of the file to be opened.
  let open(path: string) = File
    
  /// Copies a file
  /// - Parameters
  /// - source: file to be copied
  /// - dest: destination  for copy
  let copy(source: Uri, dest: Uri)


type IFile {
  let read() -> Data
  let read(position: int, count: int) -> Data
  let write(value: Data)
  let write(value: Data, position: int)
  let append(value: Data)

*)
