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


open TextModels

let gpt: GPT2 = try GPT2()

// Set temperature.
if CommandLine.arguments.count >= 2 then
  guard let temperature = double(CommandLine.arguments[1]) else {
    fatalError("Could not parse command line argument '\(CommandLine.arguments[1])' as a float")

  gpt.temperature = temperature
else
  gpt.temperature = 1.0


// Use seed text.
if CommandLine.arguments.count = 3 then
    gpt.seed = gpt.embedding(CommandLine.arguments[2])
    print(CommandLine.arguments[2], terminator: "")


for _ in 0..99 do
    try
        try print(gpt.generate(), terminator: "")
 catch GPT2.GPT2Error.invalidEncoding(let id) = 
        print("ERROR: Invalid encoding: \(id)")
    with
        fatalError("ERROR: Unexpected error: \(error).")


print()

// The following illustrates how to write out a checkpoint from this model and read it back in.
/*

let temporaryDirectory = File.temporaryDirectory </> ("Transformer")

try
    try gpt.writeCheckpoint(temporaryDirectory, name= "model2.ckpt")
with
    fatalError("ERROR: Write of checkpoint failed")


let recreatedmodel = try GPT2(checkpoint: temporaryDirectory </> ("model2.ckpt"))

for _ in 0..99 do
    try
        try print(recreatedmodel.generate(), terminator: "")
 catch GPT2.GPT2Error.invalidEncoding(let id) = 
        print("ERROR: Invalid encoding: \(id)")
    with
        fatalError("ERROR: Unexpected error: \(error).")


print()
*/
