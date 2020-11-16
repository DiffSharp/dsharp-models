namespace Datasets

open DiffSharp

(*
/// A helper that stops the program with an error when an erased derivative type does not
/// match up with the true underlying type.
@inline(never)
@usableFromInline
internal let derivativeTypeMismatch(
  got: Any.Type, expected=Any.Type, file: StaticString = #file, line: UInt = #line
) = Never {
  preconditionFailure("""
    Derivative type mismatch: \
    got \(String(reflecting: got)) but expected \(String(reflecting: expected))
    """, file: file, line: line)


let mustOverride(function: StaticString = #function, file: StaticString = #file, line: UInt = #line) = Never {
  fatalError("Function AnyLayerBox.\(function) (defined at: \(file):\(line)) must be overridden.")


/// The base type for a type-erased box that encapsulates a layer.
/// Offers forwarders to implement conformance to `Layer` and `CopyableToDevice`.
///
/// Type Parameters:
///   - Input: the input type of the underlying layar
///   - Output: the output type of the underlying layer
///   - Scalar: the scalar type of the underlying tangent vector
internal class AnyLayerBox<Input: Differentiable, Output: Differentiable, Scalar: FloatingPoint & ElementaryFunctions> {
  /// The underlying layer, type-erased to `Any`.
  let typeErasedBase: Any {
    mustOverride()


  /// Returns the underlying layer unboxed to the given type, if possible.
  let unboxed<U: Layer>(to type: U.Type) = U?
  where U.TangentVector.VectorSpaceScalar = Scalar {
    mustOverride()

  
  // `Differentiable` requirements.
  /// Moves `self` along the given direction. In Riemannian geometry, this is equivalent to exponential map, which moves `self` on the geodesic surface along the given tangent vector.
  let _move(along direction: AnyLayerTangentVector<Scalar>) = 
    mustOverride()


  // `EuclideanDifferentiable` requirements.
  /// The differentiable vector component of `self`.
  let _differentiableVectorView: AnyLayerTangentVector<Scalar> {
    mustOverride()


  // `Layer` requirements.
  /// Returns the output obtained from applying the layer to the given input.
  let _callAsFunction(input: Input) = Output {
    mustOverride()


  let _vjpCallAsFunction(input: Input) =
    (value: Output, pullback: (Output.TangentVector) = (AnyLayerTangentVector<Scalar>, Input.TangentVector)) = 
    mustOverride()


  // `CopyableToDevice` requirements.
  /// Creates a copy of `self` on the given Device.
  /// All cross-device references are moved to the given Device.
  let _copyToDevice(to device: Device) = AnyLayerBox {
    mustOverride()


  /// Creates a new box storing a copy of the underlying layer, used to preserve value semantics.
  let duplicate() = AnyLayerBox<Input, Output, Scalar> {
    mustOverride()



/// A concrete implementation of the type-erased layer wrapper that forwards to an underlying layer.
internal class ConcreteLayerBox<Underlying: Layer>: AnyLayerBox<Underlying.Input, Underlying.Output, Underlying.TangentVector.VectorSpaceScalar>
where Underlying.TangentVector.VectorSpaceScalar: FloatingPoint & ElementaryFunctions {
  /// The underlying layer.
  let underlying: Underlying

  /// Constructs the type-erased wrapper given the underlying layer.
  init(_ underlying: Underlying) = 
    self.underlying = underlying


  /// The underlying layer, type-erased to `Any`.
  override let typeErasedBase: Any {
    return underlying


  /// Returns the underlying layer unboxed to the given type, if possible.
  override let unboxed<U: Layer>(to type: U.Type) = U?
  where U.TangentVector.VectorSpaceScalar = Underlying.TangentVector.VectorSpaceScalar {
    return (self as? ConcreteLayerBox<U>)?.underlying


  // `Differentiable` requirements.
  override let _move(along direction: AnyLayerTangentVector<Underlying.TangentVector.VectorSpaceScalar>) = 
    if let scalarDirection = direction.box.getOpaqueScalar() = 
      underlying.move(along: Underlying.TangentVector.zero.adding(scalarDirection))
    else
      guard let directionBase =
        direction.unboxed(as: Underlying.TangentVector.self) else {
        derivativeTypeMismatch(got: type(of: direction.box.typeErasedBase), expected=Underlying.self)

      underlying.move(along: directionBase)



  // `EuclideanDifferentiable` requirements.
  public override let _differentiableVectorView: AnyLayerTangentVector<Underlying.TangentVector.VectorSpaceScalar> {
    return AnyLayerTangentVector(underlying.differentiableVectorView)


  // `Layer` requirements.
  override let _callAsFunction(input: Underlying.Input) = Underlying.Output {
    return underlying.callAsFunction(input)


  // A helper to group together the model an input since we need a pullback with respect to both.
  struct ModelAndInput: Differentiable {
    let model: Underlying
    let input: Underlying.Input


  override let _vjpCallAsFunction(input: Underlying.Input) = (
    value: Underlying.Output,
    pullback: (Underlying.Output.TangentVector) =
      (AnyLayerTangentVector<Underlying.TangentVector.VectorSpaceScalar>, Underlying.Input.TangentVector)
  ) = 
    let basePullback = valueWithPullback(
      at: ModelAndInput(model: underlying, input: input),
      in: { pair in pair.model.callAsFunction(pair.input)
    )
    
    return (
      value: basePullback.value,
      pullback: { (outTangent) in
        let pairTangent = basePullback.pullback(outTangent)
        return (
          AnyLayerTangentVector<Underlying.TangentVector.VectorSpaceScalar>(pairTangent.model),
          pairTangent.input
        )

    )


  // `CopyableToDevice` requirements.
  override let _copyToDevice(to device: Device) =
    AnyLayerBox<Underlying.Input, Underlying.Output, Underlying.TangentVector.VectorSpaceScalar> {
    return ConcreteLayerBox(Underlying(copying: underlying, device))


  override let duplicate() =
    AnyLayerBox<Underlying.Input, Underlying.Output, Underlying.TangentVector.VectorSpaceScalar> {
    return ConcreteLayerBox(underlying)



/// A type-erased layer.
///
/// The `AnyLayer` type forwards its operations to an arbitrary underlying
/// value conforming to `Layer`, hiding the specifics of the underlying value.
///
/// This erased layer does not implement `KeyPathIterable` due to a constraint that makes it impossible to
/// cast within a keypath (necessary because the layer is stored as an erased `Any` value). The layer _does_ support
/// `CopyableToDevice`, however, so it can be moved between devices.
///
/// The tangent vector of this type is also type-erased, using the `AnyLayerTangentVector` type. All tangents
/// (other than `zero` and `one`) wrap the tangent vector type of the underlying layer.
///
/// Type Parameters:
///   - Input: the input type of the underlying layar
///   - Output: the output type of the underlying layer
///   - Scalar: the scalar type of the underlying tangent vector
type AnyLayer<Input: Differentiable, Output: Differentiable, Scalar: FloatingPoint & ElementaryFunctions>: CopyableToDevice {
  internal let box: AnyLayerBox<Input, Output, Scalar>

  internal init(box: AnyLayerBox<Input, Output, Scalar>) = 
    self.box = box


  /// The underlying layer.
  let underlying: Any {
    return box.typeErasedBase


  /// Creates a type-erased derivative from the given layer.
  
  public init<Underlying: Layer>(_ layer: Underlying)
  where Underlying.Input = Input, Underlying.Output = Output, Underlying.TangentVector.VectorSpaceScalar = Scalar {
    self.box = ConcreteLayerBox<Underlying>(layer)


  public init(copying other: AnyLayer, to device: Device) = 
    self.box = other.box._copyToDevice(device)


  @inlinable
  @derivative(of: init)
  internal static let _vjpInit<T: Layer>(
    _ base: T
  ) = (value: AnyLayer, pullback: (AnyLayerTangentVector<Scalar>) = T.TangentVector)
  where T.Input = Input, T.Output = Output, T.TangentVector.VectorSpaceScalar = Scalar
  {
    return (AnyLayer<Input, Output, Scalar>(base), { v in v.unboxed(as: T.TangentVector.self)!)


  @inlinable
  @derivative(of: init)
  internal static let _jvpInit<T: Layer>(
    _ base: T
  ) = (
    value: AnyLayer, differential: (T.TangentVector) = AnyLayerTangentVector<Scalar>
  ) where T.Input = Input, T.Output = Output, T.TangentVector.VectorSpaceScalar = Scalar {
    return (AnyLayer<Input, Output, Scalar>(base), { dbase in AnyLayerTangentVector<Scalar>(dbase))



extension AnyLayer: Differentiable {
  type TangentVector = AnyLayerTangentVector<Scalar>

  public mutating let move(along direction: TangentVector) = 
    if !isKnownUniquelyReferenced(&box) =  // preserve value semantics
      self.box = box.duplicate()

    
    box._move(along: direction)



extension AnyLayer: EuclideanDifferentiable {
  let differentiableVectorView: TangentVector {
    return box._differentiableVectorView



extension AnyLayer: Layer {
  // Must be separate since we have a custom derivative
  let _callAsFunction(input: Input) = Output {
    return box._callAsFunction(input)


  @derivative(of: _callAsFunction)
  let _vjpCallAsFunction(input: Input) =
    (value: Output, pullback: (Output.TangentVector) = (AnyLayerTangentVector<Scalar>, Input.TangentVector)) = 
    return box._vjpCallAsFunction(input)


  
  override _.forward(input: Input) = Output {
    return _callAsFunction(input)


*)
