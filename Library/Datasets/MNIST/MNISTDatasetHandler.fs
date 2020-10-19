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

open DiffSharp
open System

type MNISTHelpers() =

    static member fetchMNISTDataset(localStorageDirectory: FilePath, remoteBaseDirectory: string, imagesFilename: string, labelsFilename: string) =
        let remoteRoot = Uri(remoteBaseDirectory) 

        let imagesData = DatasetUtilities.fetchResource(filename=imagesFilename,fileExtension="gz",remoteRoot=remoteRoot,localStorageDirectory=localStorageDirectory)
        let labelsData = DatasetUtilities.fetchResource(filename=labelsFilename, fileExtension="gz", remoteRoot=remoteRoot, localStorageDirectory=localStorageDirectory)

        let images = imagesData |> Array.skip 16
        let labels = labelsData |> Array.skip 8 |> Array.map int32
    
        let labeledImages = ResizeArray<byte[] * int32>()

        let imageByteSize = 28 * 28
        for imageIndex in 0..labels.Length-1 do
            let baseAddress = imageIndex * imageByteSize
            let data = images.[baseAddress..(baseAddress + imageByteSize)-1]
            labeledImages.Add (data, labels.[imageIndex])

        labeledImages.ToArray()
    
    static member makeMNISTBatch(samples: (byte[] * int32)[], flattening: bool, normalizing: bool, device: Device) = 
        let bytes = samples |> Seq.map fst |> Seq.reduce(Seq.toArray >> Array.append)
        let shape = if flattening then [samples.Length; 28 * 28] else [samples.Length; 28; 28; 1]
        let images = dsharp.tensor(bytes, device=device, dtype=Dtype.Byte).view(shape)
  
        let imageTensor = dsharp.tensor(images, dtype=Dtype.Float32) / 255.0
        let imageTensor = 
            if normalizing then
               imageTensor * 2.0 - 1.0
            else imageTensor
  
        let labels = dsharp.tensor(samples |> Array.map snd, device=device, dtype=Dtype.Int32)
        imageTensor, labels
