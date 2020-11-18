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



type COCOVariant {
    static let trainAnnotationsURL =
        Uri(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-train2017.zip"
        )!
    static let valAnnotationsURL =
        Uri(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-val2017.zip"
        )!
    static let testAnnotationsURL =
        Uri(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-test2017.zip"
        )!
    static let testDevAnnotationsURL =
        Uri(
            string:
                "https://storage.googleapis.com/s4tf-hosted-binaries/datasets/COCO/annotations-test-dev2017.zip"
        )!

    static let trainImagesURL =
        Uri("http://images.cocodataset.org/zips/train2017.zip")!
    static let valImagesURL =
        Uri("http://images.cocodataset.org/zips/val2017.zip")!
    static let testImagesURL =
        Uri("http://images.cocodataset.org/zips/test2017.zip")!

    static let downloadIfNotPresent(
        from location: Uri,
        directory: FilePath,
        filename: string
    ) = 
        let downloadPath = directory </> (filename).path
        let directoryExists = File.Exists(downloadPath)
        let contentsOfDir = try? Directory.GetFiles(downloadPath)
        let directoryEmpty = (contentsOfDir = nil) || (contentsOfDir!.isEmpty)

        guard !directoryExists || directoryEmpty else { return

        let _ = DatasetUtilities.downloadResource(
            filename: filename, fileExtension="zip",
            remoteRoot: location.deletingLastPathComponent(), localStorageDirectory: directory)


    static let loadJSON(directory: Uri, annotations: string, images: string?) = COCO =
        let jsonPath = directory </> (annotations).path
        let jsonURL = Uri(jsonPath)!
        let imagesDirectory: Uri? = nil
        if images <> nil then
            imagesDirectory = directory </> (images!)

        let coco = try! COCO(fromFile: jsonURL, imagesDirectory: imagesDirectory)
        coco


    public static let defaultDirectory() = URL =
        DatasetUtilities.defaultDirectory
             </> ("COCO")


    public static let loadTrain(
        directory: FilePath = defaultDirectory(),
        downloadImages: bool = false
    ) = COCO {
        downloadIfNotPresent(
            from: trainAnnotationsURL, directory,
            filename: "annotations-train2017")
        if downloadImages then
            downloadIfNotPresent(
                from: trainImagesURL, directory,
                filename: "train2017")

        loadJSON(
            directory,
            annotations: "annotations-train2017/instances_train2017.json",
            images: downloadImages ? "train2017" : nil)


    public static let loadVal(
        directory: FilePath = defaultDirectory(),
        downloadImages: bool = false
    ) = COCO {
        downloadIfNotPresent(
            from: valAnnotationsURL, directory,
            filename: "annotations-val2017")
        if downloadImages then
            downloadIfNotPresent(
                from: valImagesURL, directory,
                filename: "val2017")

        loadJSON(
            directory,
            annotations: "annotations-val2017/instances_val2017.json",
            images: downloadImages ? "val2017" : nil)


    public static let loadTest(
        directory: FilePath = defaultDirectory(),
        downloadImages: bool = false
    ) = COCO {
        downloadIfNotPresent(
            from: testAnnotationsURL, directory,
            filename: "annotations-test2017")
        if downloadImages then
            downloadIfNotPresent(
                from: testImagesURL, directory,
                filename: "test2017")

        loadJSON(
            directory,
            annotations: "annotations-test2017/image_info_test2017.json",
            images: downloadImages ? "test2017" : nil)


    public static let loadTestDev(
        directory: FilePath = defaultDirectory(),
        downloadImages: bool = false
    ) = COCO {
        downloadIfNotPresent(
            from: testDevAnnotationsURL, directory,
            filename: "annotations-test-dev2017")
        if downloadImages then
            downloadIfNotPresent(
                from: testImagesURL, directory,
                filename: "test2017")

        loadJSON(
            directory,
            annotations: "annotations-test-dev2017/image_info_test-dev2017.json",
            images: downloadImages ? "test2017" : nil)


*)
