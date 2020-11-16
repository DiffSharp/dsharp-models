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

#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Core.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\DiffSharp.Backends.ShapeChecking.dll"
#r @"..\..\bin\Debug\netcoreapp3.1\publish\Library.dll"
#r @"System.Runtime.Extensions.dll"

//open ArgumentParser
open DiffSharp

type ComplexTensor(real: Tensor, imaginary: Tensor) =
    member _.real = real
    member _.imaginary = imaginary
    static member (+)(lhs: ComplexTensor, rhs: ComplexTensor) : ComplexTensor =
      let real = lhs.real + rhs.real
      let imaginary = lhs.imaginary + rhs.imaginary
      ComplexTensor(real, imaginary)

    static member (*)(lhs: ComplexTensor, rhs: ComplexTensor) : ComplexTensor =
      let real = lhs.real * rhs.real - lhs.imaginary * rhs.imaginary
      let imaginary = lhs.real * rhs.imaginary + lhs.imaginary * rhs.real
      ComplexTensor(real, imaginary)

    static member Abs(value: ComplexTensor) : Tensor =
      value.real * value.real + value.imaginary * value.imaginary

type ComplexRegion(realMinimum: double, realMaximum: double, imaginaryMinimum: double, imaginaryMaximum: double) =

    member _.realMinimum = realMinimum
    member _.realMaximum = realMaximum
    member _.imaginaryMinimum = imaginaryMinimum
    member _.imaginaryMaximum = imaginaryMaximum
    new (argument: string) = 
        let subArguments = argument.Split(',') |> Array.map double
        ComplexRegion(subArguments.[0], subArguments.[1], subArguments.[2], subArguments.[3])

    override self.ToString() =
        $"{self.realMinimum},{self.realMaximum},{self.imaginaryMinimum},{self.imaginaryMaximum}"


