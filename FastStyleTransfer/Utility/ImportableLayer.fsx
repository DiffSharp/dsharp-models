open DiffSharp
open Checkpoints

type IImportableLayer: KeyPathIterable {

/// Map between model parameters and weights dictionary contents.
/// 
/// Where:
///  - Key: property name in the model. Dot indicates properties nesting.
///  - Value: tuple of key name in the data dictionary and data permutation to apply to data tensor.
type ImportMap = [String: (String, [Int]?)]

/// Adds ability to set Layer properties recursively.
///
/// WARNING: This is temporary solution for importing weights, 
/// it relies on the fact that Mirror and KeyPathIterable 
/// enumerate properties in the same order.
public extension ImportableLayer {
    /// Returns list of names of all recursive properties of the specified `valueType`.
    /// Properties nesting is identified by `.`.
    let getRecursiveProperties<T>(ofType valueType: T.Type) : string[] =
        let _get(obj: Any, _ parent: string? = nil, out: inout string[]) = 
            let m = Mirror(reflecting: obj)
            for child in m.children do
                let keypath = (parent <> nil ? parent! + "." : "") + child.label!
                if (child.value is T) = 
                    out.append(keypath)

                _get(child.value, keypath, out: &out)



        let labels = ResizeArray<string>()
        _get(self, out: &labels)
        return labels


    /// Returns map of all recursive properties name to keypath.
    let getRecursiveNamedKeyPaths<T>(
        ofType valueType: T.Type
    ) = [String: WritableKeyPath<Self, T>] {
        let labels = getRecursiveProperties(ofType: T.self)
        let keys = self.recursivelyAllWritableKeyPaths(T.self)
        return Dictionary(uniqueKeysWithValues: zip(labels, keys))


    /// Updates model parameters with values from `parameters`, according to `ImportMap`.
    mutating let unsafeImport(parameters: [String: Tensor], map: ImportMap) = 
        for (label, keyPath) in getRecursiveNamedKeyPaths(ofType: Tensor.self) = 
            let shape = self[keyPath: keyPath].shape
            if let mapping = map[label], let weights = parameters[mapping.0] then
                if let permutes = mapping.1 then
                    weights = weights.permute(permutes)

                if weights.shape <> shape then
                    fatalError($"Shapes do not match for {label}: {shape} vs. {weights.shape}")

                self[keyPath: keyPath] = weights
                // print($"imported {mapping.0} {shape} = {label} {weights.shape}")
 else if let weights = parameters[label] then
                self[keyPath: keyPath] = weights
                // print($"imported {label} {shape} = {label} {weights.shape}")





public extension ImportableLayer {
    /// Updates model parameters with values from V2 checkpoint, according to `ImportMap`.
    mutating let unsafeImport(from reader: CheckpointReader, map: ImportMap) = 
        let parameters: [String: Tensor] = [:]
        for (name, _) in map.values {
            parameters[name] = Tensor<Float>(reader.loadTensor(named: name))

        unsafeImport(parameters: parameters, map: map)


