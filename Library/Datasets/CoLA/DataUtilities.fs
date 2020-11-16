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

let extract(zipFileAt source: Uri, to destination: Uri) =
    print("Extracting file at '{source.path}'.")
    let process = Process()
    process.environment = ProcessInfo.processInfo.environment
    process.executableURL = Uri(fileURLWithPath= "/bin/bash")
    process.arguments = ["-c", $"unzip -d {destination.path} {source.path}"]
    try process.run()
    process.waitUntilExit()


let extract(tarGZippedFileAt source: Uri, to destination: Uri) =
    print($"Extracting file at '{source.path}'.")
    try Directory.Create(
        at: destination,
        withIntermediateDirectories: false)
    let process = Process()
    process.environment = ProcessInfo.processInfo.environment
    process.executableURL = Uri(fileURLWithPath= "/bin/bash")
    process.arguments = ["-c", $"tar -C {destination.path} -xzf {source.path}"]
    try process.run()
    process.waitUntilExit()
*)
