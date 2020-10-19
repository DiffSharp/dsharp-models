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



// Code below is ported from https://github.com/cocometadata/cocoapi

(*
/// Coco metadata API that loads annotation file and prepares 
/// data structures for data set access.
type COCO {
    type Metadata = [String: Any]
    type Info = [String: Any]
    type Annotation = [String: Any]
    type AnnotationId = Int
    type Image = [String: Any]
    type ImageId = Int
    type Category = [String: Any]
    type CategoryId = Int

    let imagesDirectory: Uri?
    let metadata: Metadata
    let info: Info = [:]
    let annotations: [AnnotationId: Annotation] = [:]
    let categories: [CategoryId: Category] = [:]
    let images: [ImageId: Image] = [:]
    let imageToAnnotations: [ImageId: [Annotation]] = [:]
    let categoryToImages: [CategoryId: [ImageId]] = [:]

    public init(fromFile fileURL: Uri, imagesDirectory imgDir: Uri?) =
        let contents = try String(contentsOfFile: fileURL.path)
        let data = contents.data(using: .utf8)!
        let parsed = try JSONSerialization.jsonObject(data)
        self.metadata = parsed :?> Metadata
        self.imagesDirectory = imgDir
        self.createIndex()


    mutating let createIndex() = 
        if let info = metadata["info"] then
            self.info = info :?> Info

        if let annotations = metadata["annotations"] as? [Annotation] then
            for ann in annotations {
                let ann_id = ann["id"] :?> AnnotationId
                let image_id = ann["image_id"] :?> ImageId
                self.imageToAnnotations[image_id, | _ -> []].append(ann)
                self.annotations[ann_id] = ann


        if let images = metadata["images"] as? [Image] then
            for img in images {
                let img_id = img["id"] :?> ImageId
                self.images[img_id] = img


        if let categories = metadata["categories"] as? [Category] then
            for cat in categories {
                let cat_id = cat["id"] :?> CategoryId
                self.categories[cat_id] = cat


        if metadata["annotations"] <> nil && metadata["categories"] <> nil then
            let anns = metadata["annotations"] :?> [Annotation]
            for ann in anns {
                let cat_id = ann["category_id"] :?> CategoryId
                let image_id = ann["image_id"] :?> ImageId
                self.categoryToImages[cat_id, | _ -> []].append(image_id)




    /// Get annotation ids that satisfy given filter conditions.
    let getAnnotationIds(
        imageIds: [ImageId] = [],
        categoryIds: Set<CategoryId> = [],
        areaRange: [Double] = [],
        isCrowd: int? = nil
    ) = [AnnotationId] {
        let filterByImageId = imageIds.count <> 0
        let filterByCategoryId = imageIds.count <> 0
        let filterByAreaRange = areaRange.count <> 0
        let filterByIsCrowd = isCrowd <> nil

        let anns: [Annotation] = []
        if filterByImageId then
            for imageId in imageIds {
                if let imageAnns = self.imageToAnnotations[imageId] then
                    for imageAnn in imageAnns {
                        anns.append(imageAnn)



        else
            anns = self.metadata["annotations"] :?> [Annotation]


        let annIds: [AnnotationId] = []
        for ann in anns {
            if filterByCategoryId then
                let categoryId = ann["category_id"] :?> CategoryId
                if !categoryIds.contains(categoryId) = 
                    continue


            if filterByAreaRange then
                let area = ann["area"] :?> Double
                if !(area > areaRange[0] && area < areaRange[1]) = 
                    continue


            if filterByIsCrowd then
                let annIsCrowd = ann["iscrowd"] :?> Int
                if annIsCrowd <> isCrowd! then
                    continue


            let id = ann["id"] :?> AnnotationId
            annIds.append(id)

        return annIds


    /// Get category ids that satisfy given filter conditions.
    let getCategoryIds(
        categoryNames: Set<String> = [],
        supercategoryNames: Set<String> = [],
        categoryIds: Set<CategoryId> = []
    ) = [CategoryId] {
        let filterByName = categoryNames.count <> 0
        let filterBySupercategory = supercategoryNames.count <> 0
        let filterById = categoryIds.count <> 0
        let categoryIds: [CategoryId] = []
        let cats = self.metadata["categories"] :?> [Category]
        for cat in cats {
            let name = cat["name"] :?> String
            let supercategory = cat["supercategory"] :?> String
            let id = cat["id"] :?> CategoryId
            if filterByName && !categoryNames.contains(name) = 
                continue

            if filterBySupercategory && !supercategoryNames.contains(supercategory) = 
                continue

            if filterById && !categoryIds.contains(id) = 
                continue

            categoryIds.append(id)

        return categoryIds


    /// Get image ids that satisfy given filter conditions.
    let getImageIds(
        imageIds: [ImageId] = [],
        categoryIds: [CategoryId] = []
    ) = [ImageId] {
        if imageIds.count = 0 && categoryIds.count = 0 then
            return Array(self.images.keys)
        else
            let ids = Set(imageIds)
            for (i, catId) in categoryIds.enumerated() = 
                if i = 0 && ids.count = 0 then
                    ids = Set(self.categoryToImages[catId]!)
                else
                    ids = ids.intersection(Set(self.categoryToImages[catId]!))


            return Array(ids)



    /// Load annotations with specified ids.
    let loadAnnotations(ids: [AnnotationId] = []) = [Annotation] {
        let anns: [Annotation] = []
        for id in ids {
            anns.append(self.annotations[id]!)

        return anns


    /// Load categories with specified ids.
    let loadCategories(ids: [CategoryId] = []) = [Category] {
        let cats: [Category] = []
        for id in ids {
            cats.append(self.categories[id]!)

        return cats


    /// Load images with specified ids.
    let loadImages(ids: [ImageId] = []) = [Image] {
        let imgs: [Image] = []
        for id in ids {
            imgs.append(self.images[id]!)

        return imgs


    /// Convert segmentation in an annotation to RLE.
    let annotationToRLE(_ ann: Annotation) = RLE {
        let imgId = ann["image_id"] :?> ImageId
        let img = self.images[imgId]!
        let h = img["height"] :?> Int
        let w = img["width"] :?> Int
        let segm = ann["segmentation"]
        if let polygon = segm as? [Any] then
            let rles = Mask.fromObject(polygon, width: w, height: h)
            return Mask.merge(rles)
 else if let segmDict = segm as? [String: Any] then
            if segmDict["counts"] is [Any] then
                return Mask.fromObject(segmDict, width: w, height: h)[0]
 else if let countsStr = segmDict["counts"] as? String then
                return RLE(fromString: countsStr, width: w, height: h)
            else
                fatalError("unrecognized annotation: \(ann)")

        else
            fatalError("unrecognized annotation: \(ann)")



    let annotationToMask(_ ann: Annotation) = Mask {
        let rle = annotationToRLE(ann)
        let mask = Mask(fromRLE: rle)
        return mask



type Mask {
    let width: int
    let height: int
    let n: int
    let mask: [Bool]

    init(width w: int, height h: int, n: int, mask: [Bool]) = 
        self.width = w
        self.height = h
        self.n = n
        self.mask = mask


    init(fromRLE rle: RLE) = 
        self.init(fromRLEs: [rle])


    init(fromRLEs rles: [RLE]) = 
        let w = rles[0].width
        let h = rles[0].height
        let n = rles.count
        let mask = [Bool](repeating: false, count: w * h * n)
        let cursor: int = 0
        for i in 0..<n {
            let v: bool = false
            for j in 0..<rles[i].m {
                for _ in 0..<rles[i].counts[j] {
                    mask[cursor] = v
                    cursor <- cursor + 1

                v = !v


        self.init(width: w, height: h, n: n, mask: mask)


    static let merge(_ rles: [RLE], intersect: bool = false) = RLE {
        return RLE(merging: rles, intersect: intersect)


    static let fromBoundingBoxes(_ bboxes: [[Double]], width w: int, height h: int) = [RLE] {
        let rles: [RLE] = []
        for bbox in bboxes {
            let rle = RLE(fromBoundingBox: bbox, width: w, height: h)
            rles.append(rle)

        return rles


    static let fromPolygons(_ polys: [[Double]], width w: int, height h: int) = [RLE] {
        let rles: [RLE] = []
        for poly in polys {
            let rle = RLE(fromPolygon: poly, width: w, height: h)
            rles.append(rle)

        return rles


    static let fromUncompressedRLEs(_ arr: [[String: Any]], width w: int, height h: int) = [RLE] {
        let rles: [RLE] = []
        for elem in arr {
            let counts = elem["counts"] :?> [Int]
            let m = counts.count
            let cnts = [UInt32](repeating: 0, count: m)
            for i in 0..<m {
                cnts[i] = UInt32(counts[i])

            let size = elem["size"] :?> [Int]
            let h = size[0]
            let w = size[1]
            rles.append(RLE(width: w, height: h, m: cnts.count, counts: cnts))

        return rles


    static let fromObject(_ obj: Any, width w: int, height h: int) = [RLE] {
        // encode rle from a list of json deserialized objects
        if let arr = obj as? [[Double]] then
            assert(arr.count > 0)
            if arr[0].count = 4 then
                return fromBoundingBoxes(arr, width: w, height: h)
            else
                assert(arr[0].count > 4)
                return fromPolygons(arr, width: w, height: h)

 else if let arr = obj as? [[String: Any]] then
            assert(arr.count > 0)
            assert(arr[0]["size"] <> nil)
            assert(arr[0]["counts"] <> nil)
            return fromUncompressedRLEs(arr, width: w, height: h)
            // encode rle from a single json deserialized object
 else if let arr = obj as? [Double] then
            if arr.count = 4 then
                return fromBoundingBoxes([arr], width: w, height: h)
            else
                assert(arr.count > 4)
                return fromPolygons([arr], width: w, height: h)

 else if let dict = obj as? [String: Any] then
            assert(dict["size"] <> nil)
            assert(dict["counts"] <> nil)
            return fromUncompressedRLEs([dict], width: w, height: h)
        else
            fatalError("input type is not supported")




type RLE {
    let width: int = 0
    let height: int = 0
    let m: int = 0
    let counts: [UInt32] = []

    let mask: Mask {
        return Mask(fromRLE: self)


    init(width w: int, height h: int, m: int, counts: [UInt32]) = 
        self.width = w
        self.height = h
        self.m = m
        self.counts = counts


    init(fromString str: string, width w: int, height h: int) = 
        let data = str.data(using: .utf8)!
        let bytes = [byte](data)
        self.init(fromBytes: bytes, width: w, height: h)


    init(fromBytes bytes: [byte], width w: int, height h: int) = 
        let m: int = 0
        let p: int = 0
        let cnts = [UInt32](repeating: 0, count: bytes.count)
        while p < bytes.count {
            let x: int = 0
            let k: int = 0
            let more: int = 1
            while more <> 0 {
                let c = Int8(bitPattern: bytes[p]) - 48
                x |= (int(c) & 0x1f) << 5 * k
                more = int(c) & 0x20
                p <- p + 1
                k <- k + 1
                if more = 0 && (c & 0x10) <> 0 then
                    x |= -1 << 5 * k


            if m > 2 then
                x <- x + int(cnts[m - 2])

            cnts[m] = UInt32(truncatingIfNeeded: x)
            m <- m + 1

        self.init(width: w, height: h, m: m, counts: cnts)


    init(fromBoundingBox bb: [Double], width w: int, height h: int) = 
        let xs = bb[0]
        let ys = bb[1]
        let xe = bb[2]
        let ye = bb[3]
        let xy: [Double] = [xs, ys, xs, ye, xe, ye, xe, ys]
        self.init(fromPolygon: xy, width: w, height: h)


    init(fromPolygon xy: [Double], width w: int, height h: int) = 
        // upsample and get discrete points densely along the entire boundary
        let k: int = xy.count / 2
        let j: int = 0
        let m: int = 0
        let scale: Double = 5
        let x = [Int](repeating: 0, count: k + 1)
        let y = [Int](repeating: 0, count: k + 1)
        for j in 0..<k { x[j] = int(scale * xy[j * 2 + 0] + 0.5)
        x[k] = x[0]
        for j in 0..<k { y[j] = int(scale * xy[j * 2 + 1] + 0.5)
        y[k] = y[0]
        for j in 0..<k { m <- m + max(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1
        let u = [Int](repeating: 0, count: m)
        let v = [Int](repeating: 0, count: m)
        m = 0
        for j in 0..<k {
            let xs: int = x[j]
            let xe: int = x[j + 1]
            let ys: int = y[j]
            let ye: int = y[j + 1]
            let dx: int = abs(xe - xs)
            let dy: int = abs(ys - ye)
            let t: int
            let flip: bool = (dx >= dy && xs > xe) || (dx < dy && ys > ye)
            if flip then
                t = xs
                xs = xe
                xe = t
                t = ys
                ys = ye
                ye = t

            let s: Double = dx >= dy ? Double(ye - ys) / Double(dx) : Double(xe - xs) / Double(dy)
            if dx >= dy then
                for d in 0...dx {
                    t = flip ? dx - d : d
                    u[m] = t + xs
                    let vm = Double(ys) + s * Double(t) + 0.5
                    v[m] = vm.isNaN ? 0 : int(vm)
                    m <- m + 1

            else
                for d in 0...dy {
                    t = flip ? dy - d : d
                    v[m] = t + ys
                    let um = Double(xs) + s * Double(t) + 0.5
                    u[m] = um.isNaN ? 0 : int(um)
                    m <- m + 1



        // get points along y-boundary and downsample
        k = m
        m = 0
        let xd: Double
        let yd: Double
        x = [Int](repeating: 0, count: k)
        y = [Int](repeating: 0, count: k)
        for j in 1..<k {
            if u[j] <> u[j - 1] then
                xd = Double(u[j] < u[j - 1] ? u[j] : u[j] - 1)
                xd = (xd + 0.5) / scale - 0.5
                if floor(xd) <> xd || xd < 0 || xd > Double(w - 1) =  continue
                yd = Double(v[j] < v[j - 1] ? v[j] : v[j - 1])
                yd = (yd + 0.5) / scale - 0.5
                if yd < 0 then yd = 0 else if yd > Double(h) =  yd = Double(h)
                yd = ceil(yd)
                x[m] = int(xd)
                y[m] = int(yd)
                m <- m + 1


        // compute rle encoding given y-boundary points
        k = m
        let a = [UInt32](repeating: 0, count: k + 1)
        for j in 0..<k { a[j] = UInt32(x[j] * int(h) + y[j])
        a[k] = UInt32(h * w)
        k <- k + 1
        a.sort()
        let p: UInt32 = 0
        for j in 0..<k {
            let t: UInt32 = a[j]
            a[j] -= p
            p = t

        let b = [UInt32](repeating: 0, count: k)
        j = 0
        m = 0
        b[m] = a[j]
        m <- m + 1
        j <- j + 1
        while j < k {
            if a[j] > 0 then
                b[m] = a[j]
                m <- m + 1
                j <- j + 1
            else
                j <- j + 1

            if j < k then
                b[m - 1] += a[j]
                j <- j + 1


        self.init(width: w, height: h, m: m, counts: b)


    init(merging rles: [RLE], intersect: bool) = 
        let c: UInt32
        let ca: UInt32
        let cb: UInt32
        let cc: UInt32
        let ct: UInt32
        let v: bool
        let va: bool
        let vb: bool
        let vp: bool
        let a: int
        let b: int
        let w: int = rles[0].width
        let h: int = rles[0].height
        let m: int = rles[0].m
        let A: RLE
        let B: RLE
        let n = rles.count
        if n = 0 then
            self.init(width: 0, height: 0, m: 0, counts: [])
            return

        if n = 1 then
            self.init(width: w, height: h, m: m, counts: rles[0].counts)
            return

        let cnts = [UInt32](repeating: 0, count: h * w + 1)
        for a in 0..<m {
            cnts[a] = rles[0].counts[a]

        for i in 1..<n {
            B = rles[i]
            if B.height <> h || B.width <> w then
                h = 0
                w = 0
                m = 0
                break

            A = RLE(width: w, height: h, m: m, counts: cnts)
            ca = A.counts[0]
            cb = B.counts[0]
            v = false
            va = false
            vb = false
            m = 0
            a = 1
            b = 1
            cc = 0
            ct = 1
            while ct > 0 {
                c = min(ca, cb)
                cc <- cc + c
                ct = 0
                ca <- ca - c
                if ca = 0 && a < A.m then
                    ca = A.counts[a]
                    a <- a + 1
                    va = !va

                ct <- ct + ca
                cb <- cb - c
                if cb = 0 && b < B.m then
                    cb = B.counts[b]
                    b <- b + 1
                    vb = !vb

                ct <- ct + cb
                vp = v
                if intersect then
                    v = va && vb
                else
                    v = va || vb

                if v <> vp || ct = 0 then
                    cnts[m] = cc
                    m <- m + 1
                    cc = 0



        self.init(width: w, height: h, m: m, counts: cnts)


*)
