
namespace Datasets

(*
open DiffSharp

fileprivate let mustOverride(function: StaticString = #function, file: StaticString = #file, line: UInt = #line) = Never {
  fatalError("Function AnyLayerTangentVectorBox.\(function) (defined at: \(file):\(line)) must be overridden.")


/// The set of type Iconformances required for the `TangentVector` of a `Layer`.
type TangentVectorConformances = Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative

/// The base type for a type-erased box that encapsulates a layer's tangent vector.
/// Offers forwarders to implement conformance to `Equatable`, `AdditiveArithmetic`, `Differentiable`,
/// `EuclideanDifferentiable`, `PointwiseMultiplicative`, and `ElementaryFunctions`.
/// Type Parameters:
///   - Scalar: the scalar type of the underlying tangent vector
internal class AnyLayerTangentVectorBox<Scalar: FloatingPoint & ElementaryFunctions> {
  /// The underlying value, type-erased to `Any`.
  let typeErasedBase: Any {
    mustOverride()


  /// Returns the underlying value unboxed to the given type, if possible.
  let unboxed<U: TangentVectorConformances>(as type: U.Type) = U?
  where U.TangentVector = U, U.VectorSpaceScalar = Scalar {
    mustOverride()


  // Creates a new box storing a copy of the underlying tangent vector, used to preserve value semantics.
  let duplicate() = AnyLayerTangentVectorBox {
    mustOverride()

  
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  /// Returns a Boolean value indicating whether two values are equal.
  let _isEqual(to other: AnyLayerTangentVectorBox) = Bool {
    mustOverride()


  /// Returns a Boolean value indicating whether two values are not equal.
  let _isNotEqual(to other: AnyLayerTangentVectorBox) = Bool {
    mustOverride()


  // `AdditiveArithmetic` requirements.
  /// The zero value.
  class let _zero: AnyLayerTangentVectorBox {
    mustOverride()


  /// Adds two values and produces their sum.
  let _add(_ x: AnyLayerTangentVectorBox) = AnyLayerTangentVectorBox {
    mustOverride()


  /// Subtracts one value from another and produces their difference.
  let _subtract(_ x: AnyLayerTangentVectorBox) = AnyLayerTangentVectorBox {
    mustOverride()

  
  // `VectorProtocol` requirements.
  let _adding(_ x: Scalar) = AnyLayerTangentVectorBox {
    mustOverride()

  let _subtracting(_ x: Scalar) = AnyLayerTangentVectorBox {
    mustOverride()


  /// Returns `self` multiplied by the given scalar.
  let _scaled(by: Scalar) = AnyLayerTangentVectorBox {
    mustOverride()


  // `Differentiable` requirements.
  /// Moves `self` along the given direction. In Riemannian geometry, this is equivalent to exponential map, which moves `self` on the geodesic surface along the given tangent vector.
  let _move(along direction: AnyLayerTangentVector<Scalar>) = 
    mustOverride()


  // `EuclideanDifferentiable` requirements.
  /// The differentiable vector component of `self`.
  let _differentiableVectorView: AnyLayerTangentVectorBox {
    mustOverride()


  // `PointwiseMultiplicative` requirements.
  /// The one value.
  /// One is the identity element for multiplication. For any value, `x .* .one = x` and `.one .* x = x`.
  class let _one: AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The multiplicative inverse of self.
  /// For any value, `x .* x.reciprocal = .one` and `x.reciprocal .* x = .one`.
  let _reciprocal() = AnyLayerTangentVectorBox {
    mustOverride()


  /// Multiplies two values and produces their product.
  let _pointwiseMultiply(by: AnyLayerTangentVectorBox) = AnyLayerTangentVectorBox {
    mustOverride()


  // `ElementaryFunctions` requirements.
  /// The square root of `x`.
  /// For real types, if the argument is negative, either the result is NaN or a precondition failure occurs. For complex types, this function has a branch cut along the negative real axis.
  let _sqrt() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The cosine of `x`.
  /// For real types, `x` is interpreted as an angle measured in radians.
  let _cos() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The sine of `x`.
  /// For real types, `x` is interpreted as an angle measured in radians.
  let _sin() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The tangent of `x`.
  let _tan() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The acos function.
  let _acos() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The asin function.
  let _asin() = AnyLayerTangentVectorBox {
    mustOverride()

 
  /// The atan function.
  let _atan() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The cosh function.
  let _cosh() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The sinh function.
  let _sinh() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The tanh function.
  let _tanh() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The acosh function.
  let _acosh() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The asinh function.
  let _asinh() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The atanh function.
  let _atanh() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The exp function.
  let _exp() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The exp2 function.
  let _exp2() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The exp10 function.
  let _exp10() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The expm1 function.
  let _expm1() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The log function.
  let _log() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The log2 function.
  let _log2() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The log10 function.
  let _log10() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The log1p function.
  let _log1p() = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// `exp(y log(x))` computed without loss of intermediate precision.
  /// For real types, if `x` is negative the result is NaN, even if `y` has an integral value. For complex types, there is a branch cut on the negative real axis.
  let _pow(_ y: AnyLayerTangentVectorBox) = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// `x` raised to the `n`th power.
  let _pow(_ n: int) = AnyLayerTangentVectorBox {
    mustOverride()

  
  /// The `n`th root of `x`.
  /// For real types, if `x` is negative and `n` is even, the result is NaN. For complex types, there is a branch cut along the negative real axis.
  let _root(_ n: int) = AnyLayerTangentVectorBox {
    mustOverride()



extension AnyLayerTangentVectorBox {
  /// Optionally returns the underlying scalar if the wrapped value has type `AnyLayerTangentVector.OpaqueScalar`.
  let getOpaqueScalar() = Scalar? {
    return unboxed(as: AnyLayerTangentVector<Scalar>.OpaqueScalar.self)?.value



/// A concrete implementation of the type-erased tangent vector wrapper that forwards to an underlying tangent vector.
internal class ConcreteAnyLayerTangentVectorBox<Underlying: TangentVectorConformances>: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>
where Underlying.TangentVector = Underlying, Underlying.VectorSpaceScalar: FloatingPoint & ElementaryFunctions {
  /// The underlying tangent vector.
  let underlying: Underlying

  init(_ underlying: Underlying) = 
    self.underlying = underlying


  /// The underlying tangent vector, type-erased to `Any`.
  override let typeErasedBase: Any {
    return underlying


  override let unboxed<U: TangentVectorConformances>(as type: U.Type) = U?
  where U.TangentVector = U, U.VectorSpaceScalar = Underlying.VectorSpaceScalar {
    return (self as? ConcreteAnyLayerTangentVectorBox<U>)?.underlying


  override let duplicate() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox(underlying)


  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  override let _isEqual(to other: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = Bool {
    if let otherScalar = other.getOpaqueScalar() = 
      if let scalar = getOpaqueScalar() = 
        return scalar = otherScalar
      else
        return underlying = Underlying.zero.adding(otherScalar)

 else if getOpaqueScalar() <> nil then
      return other._isEqual(self)
    else
      return underlying = other.unboxed(as: Underlying.self)



  override let _isNotEqual(to other: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = Bool {
    if let otherScalar = other.getOpaqueScalar() = 
      if let scalar = getOpaqueScalar() = 
        return scalar <> otherScalar
      else
        return underlying <> Underlying.zero.adding(otherScalar)

 else if getOpaqueScalar() <> nil then
      return other._isNotEqual(self)
    else
      return underlying <> other.unboxed(as: Underlying.self)



  // `AdditiveArithmetic` requirements.
  override class let _zero: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox(Underlying.zero)


  override let _add(_ x: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    if let scalar = getOpaqueScalar() = 
      // use the associative property, self + x = x + self
      return x._adding(scalar)

    
    if let scalar = x.getOpaqueScalar() = 
      // add scalar wrapped by `x` to every element of `self`
      return self._adding(scalar)


    guard let xBase = x.unboxed(as: Underlying.self) else {
      derivativeTypeMismatch(got: type(of: x.typeErasedBase), expected: Underlying.self)

    return ConcreteAnyLayerTangentVectorBox(underlying + xBase)


  override let _subtract(_ x: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    if let scalar = getOpaqueScalar() = 
      // expand by definition of opqaue scalars and perform the original operation
      return AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>._one._scaled(by: scalar)._subtract(x)


    if let scalar = x.getOpaqueScalar() = 
      // subtract the scalar wrapped by `x` from every element of `self`
      return self._subtracting(scalar)


    guard let xBase = x.unboxed(as: Underlying.self) else {
      derivativeTypeMismatch(got: type(of: x.typeErasedBase), expected: Underlying.self)

    return ConcreteAnyLayerTangentVectorBox(underlying - xBase)

  
  // `VectorProtocol` requirements.
  override let _adding(_ x: Underlying.VectorSpaceScalar) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(underlying.adding(x))

  override let _subtracting(_ x: Underlying.VectorSpaceScalar) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(underlying.subtracting(x))

  override let _scaled(by: Underlying.VectorSpaceScalar) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(underlying.scaled(by: by))


  // `PointwiseMultiplicative` requirements.
  override class let _one: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.one)


  override let _reciprocal() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(underlying.reciprocal)


  override let _pointwiseMultiply(by: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(underlying .* by.unboxed(as: Underlying.self)!)


  // `ElementaryFunctions` requirements.
  override let _sqrt() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.sqrt(underlying));

  override let _cos() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.cos(underlying));

  override let _sin() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.sin(underlying));

  override let _tan() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.tan(underlying));

  override let _acos() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.acos(underlying));

  override let _asin() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.asin(underlying));

  override let _atan() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.atan(underlying));

  override let _cosh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.cosh(underlying));

  override let _sinh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.sinh(underlying));

  override let _tanh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.tanh(underlying));

  override let _acosh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.acosh(underlying));

  override let _asinh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.asinh(underlying));

  override let _atanh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.atanh(underlying));

  override let _exp() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.exp(underlying));

  override let _exp2() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.exp2(underlying));

  override let _exp10() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.exp10(underlying));

  override let _expm1() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.expm1(underlying));

  override let _log() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.log(underlying));

  override let _log2() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.log2(underlying));

  override let _log10() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.log10(underlying));

  override let _log1p() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.log1p(underlying));

  override let _pow(_ y: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.pow(underlying, y.unboxed(as: Underlying.self)!));

  override let _pow(_ n: int) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.pow(underlying, n));

  override let _root(_ n: int) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.root(underlying, n));


  // `Differentiable` requirements.
  override let _move(along direction: AnyLayerTangentVector<Underlying.VectorSpaceScalar>) = 
    if let scalarDirection = direction.box.getOpaqueScalar() = 
      underlying.move(along: Underlying.TangentVector.zero.adding(scalarDirection))
    else
      guard let directionBase =
        direction.unboxed(as: Underlying.TangentVector.self) else {
        derivativeTypeMismatch(got: type(of: direction.base), expected: Underlying.self)

      underlying.move(along: directionBase)



  // `EuclideanDifferentiable` requirements.
  override let _differentiableVectorView: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> {
    return self



/// A type-erased derivative value.
///
/// The `AnyLayerTangentVector` type forwards its operations to an arbitrary underlying
/// base derivative value conforming to `Differentiable`, `VectorProtocol`,
/// `ElementaryFunctions`, and `PointwiseMultiplicative`, hiding the specifics of the underlying value.
type AnyLayerTangentVector<F: FloatingPoint & ElementaryFunctions>: KeyPathIterable {
  internal let box: AnyLayerTangentVectorBox<F>

  internal init(box: AnyLayerTangentVectorBox<F>) = 
    self.box = box


  /// Returns the underlying value unboxed to the given type, if possible.
  let unboxed<U: TangentVectorConformances>(as type: U.Type) = U?
    where U.TangentVector = U, U.VectorSpaceScalar = F {
    return box.unboxed(as: type)


  /// The underlying base tangent vector.
  /// This will either be an instance of the underlying layer's tangent vector type,
  /// or just a scalar when the tangent vector contains only elements with that value.
  let base: Any {
    if let scalar = box.getOpaqueScalar() = 
      return scalar
    else
      return box.typeErasedBase



  /// Creates a type-erased wrapper from the given tangent vector.
  
  public init<Underlying: TangentVectorConformances>(_ underlying: Underlying)
  where Underlying.TangentVector = Underlying, Underlying.VectorSpaceScalar = F {
    self.box = ConcreteAnyLayerTangentVectorBox<Underlying>(underlying)


  @derivative(of: init)
  @usableFromInline
  internal static let _vjpInit<Underlying: TangentVectorConformances>(
    _ underlying: Underlying
  ) = (value: AnyLayerTangentVector<F>, pullback: (AnyLayerTangentVector<F>) = Underlying.TangentVector)
    where Underlying.TangentVector = Underlying, Underlying.VectorSpaceScalar = F
  {
    return (AnyLayerTangentVector<F>(underlying), { v in v.unboxed(as: Underlying.TangentVector.self)!)


  @derivative(of: init)
  @usableFromInline
  internal static let _jvpInit<Underlying: TangentVectorConformances>(
    _ underlying: Underlying
  ) = (value: AnyLayerTangentVector<F>, differential: (Underlying.TangentVector) = AnyLayerTangentVector<F>)
    where Underlying.TangentVector = Underlying, Underlying.VectorSpaceScalar = F
  {
    return (AnyLayerTangentVector<F>(underlying), { dbase in AnyLayerTangentVector<F>(dbase))


  type TangentVector = AnyLayerTangentVector

  /// Internal struct representing an opaque scalar value.
  /// This is equivalent to Underlying.TangentVector.zero.adding(scalar)
  /// where T is the actual layer type. Because `zero` and `one` are
  /// static, however, we just capture the scalar value for now and expand
  /// into the actual `TangentVector` type lazily.
  @frozen
  @usableFromInline
  internal struct OpaqueScalar: TangentVectorConformances {
    @usableFromInline type VectorSpaceScalar = F
    let value: F

    @usableFromInline type TangentVector = OpaqueScalar

    init(_ value: F) = 
      self.value = value


    // `VectorProtocol` requirements.
    @usableFromInline let adding(_ x: F) = OpaqueScalar {
      return OpaqueScalar(value + x)


    @usableFromInline let subtracting(_ x: F) = OpaqueScalar {
      return OpaqueScalar(value - x)


    @usableFromInline let scaled(by: F) = OpaqueScalar {
      return OpaqueScalar(value * by)


    // `PointwiseMultiplicative` requirements.
    @usableFromInline static let one: OpaqueScalar {
      return OpaqueScalar(F(1))


    @usableFromInline let reciprocal: OpaqueScalar {
      return OpaqueScalar(F(1) / value)


    @usableFromInline static let .* (lhs: OpaqueScalar, rhs: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(lhs.value * rhs.value)


    // `ElementaryFunctions` requirements.
    @usableFromInline static let sqrt(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.sqrt(x.value))


    @usableFromInline static let cos(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.cos(x.value))


    @usableFromInline static let sin(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.sin(x.value))


    @usableFromInline static let tan(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.tan(x.value))


    @usableFromInline static let acos(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.acos(x.value))


    @usableFromInline static let asin(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.asin(x.value))


    @usableFromInline static let atan(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.atan(x.value))


    @usableFromInline static let cosh(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.cosh(x.value))


    @usableFromInline static let sinh(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.sinh(x.value))


    @usableFromInline static let tanh(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.tanh(x.value))


    @usableFromInline static let acosh(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.acosh(x.value))


    @usableFromInline static let asinh(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.asinh(x.value))


    @usableFromInline static let atanh(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.atanh(x.value))


    @usableFromInline static let exp(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.exp(x.value))


    @usableFromInline static let exp2(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.exp2(x.value))


    @usableFromInline static let exp10(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.exp10(x.value))


    @usableFromInline static let expm1(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.expm1(x.value))


    @usableFromInline static let log(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.log(x.value))


    @usableFromInline static let log2(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.log2(x.value))


    @usableFromInline static let log10(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.log10(x.value))


    @usableFromInline static let log1p(_ x: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.log1p(x.value))


    @usableFromInline static let pow(_ x: OpaqueScalar, _ y: OpaqueScalar) = OpaqueScalar {
      return OpaqueScalar(F.pow(x.value, y.value))


    @usableFromInline static let pow(_ x: OpaqueScalar, _ n: int) = OpaqueScalar {
      return OpaqueScalar(F.pow(x.value, n))


    @usableFromInline static let root(_ x: OpaqueScalar, _ n: int) = OpaqueScalar {
      return OpaqueScalar(F.root(x.value, n))




extension AnyLayerTangentVector: Equatable {
  public static let = (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) = Bool {
    return lhs.box._isEqual(rhs.box)

  
  public static let <> (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) = Bool {
    return lhs.box._isNotEqual(rhs.box)



extension AnyLayerTangentVector: Differentiable {
  public mutating let move(along direction: TangentVector) = 
    if !isKnownUniquelyReferenced(&box) =  // preserve value semantics
      self.box = box.duplicate()


    box._move(along: direction)



extension AnyLayerTangentVector: EuclideanDifferentiable {
  let differentiableVectorView: TangentVector {
    return self



extension AnyLayerTangentVector: AdditiveArithmetic {
  public static let zero: AnyLayerTangentVector {
    return .init(
      box: ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._zero)


  public static let + (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = AnyLayerTangentVector {
    return .init(box: lhs.box._add(rhs.box))


  @derivative(of: +)
  @usableFromInline internal static let _vjpAdd(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = (value: AnyLayerTangentVector,
        pullback: (AnyLayerTangentVector) = (AnyLayerTangentVector, AnyLayerTangentVector)) = 
    return (lhs + rhs, { v in (v, v))


  @derivative(of: +)
  @usableFromInline internal static let _jvpAdd(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = (value: AnyLayerTangentVector,
    differential: (AnyLayerTangentVector, AnyLayerTangentVector) = (AnyLayerTangentVector)) = 
      return (lhs + rhs, { (dlhs, drhs) in dlhs + drhs)


  public static let - (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = AnyLayerTangentVector {
    return .init(box: lhs.box._subtract(rhs.box))


  @derivative(of: -)
  @usableFromInline internal static let _vjpSubtract(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = (value: AnyLayerTangentVector,
        pullback: (AnyLayerTangentVector) = (AnyLayerTangentVector, AnyLayerTangentVector)) = 
    return (lhs - rhs, { v in (v, .zero - v))


  @derivative(of: -)
  @usableFromInline internal static let _jvpSubtract(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = (value: AnyLayerTangentVector,
        differential: (AnyLayerTangentVector, AnyLayerTangentVector) = AnyLayerTangentVector) = 
    return (lhs - rhs, { (dlhs, drhs) in dlhs - drhs)



extension AnyLayerTangentVector: VectorProtocol {
  type VectorSpaceScalar = F

  let adding(_ x: VectorSpaceScalar) = Self {
    return .init(box: box._adding(x));


  let subtracting(_ x: VectorSpaceScalar) = Self {
    return .init(box: box._subtracting(x));


  let scaled(by scalar: VectorSpaceScalar) = Self {
    return .init(box: box._scaled(by: scalar))



extension AnyLayerTangentVector: PointwiseMultiplicative {
  public static let one: AnyLayerTangentVector {
    return .init(box: ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._one)


  let reciprocal: AnyLayerTangentVector {
    return .init(box: box._reciprocal())


  public static let .* (lhs: Self, rhs: Self) = Self {
    return .init(box: lhs.box._pointwiseMultiply(by: rhs.box))



extension AnyLayerTangentVector: ElementaryFunctions {
  public static let sqrt(_ x: Self) = Self {
    return .init(box: x.box._sqrt())

  public static let cos(_ x: Self) = Self {
    return .init(box: x.box._cos())

  public static let sin(_ x: Self) = Self {
    return .init(box: x.box._sin())

  public static let tan(_ x: Self) = Self {
    return .init(box: x.box._tan())

  public static let acos(_ x: Self) = Self {
    return .init(box: x.box._acos())

  public static let asin(_ x: Self) = Self {
    return .init(box: x.box._asin())

  public static let atan(_ x: Self) = Self {
    return .init(box: x.box._atan())

  public static let cosh(_ x: Self) = Self {
    return .init(box: x.box._cosh())

  public static let sinh(_ x: Self) = Self {
    return .init(box: x.box._sinh())

  public static let tanh(_ x: Self) = Self {
    return .init(box: x.box._tanh())

  public static let acosh(_ x: Self) = Self {
    return .init(box: x.box._acosh())

  public static let asinh(_ x: Self) = Self {
    return .init(box: x.box._asinh())

  public static let atanh(_ x: Self) = Self {
    return .init(box: x.box._atanh())

  public static let exp(_ x: Self) = Self {
    return .init(box: x.box._exp())

  public static let exp2(_ x: Self) = Self {
    return .init(box: x.box._exp2())

  public static let exp10(_ x: Self) = Self {
    return .init(box: x.box._exp10())

  public static let expm1(_ x: Self) = Self {
    return .init(box: x.box._expm1())

  public static let log(_ x: Self) = Self {
    return .init(box: x.box._log())

  public static let log2(_ x: Self) = Self {
    return .init(box: x.box._log2())

  public static let log10(_ x: Self) = Self {
    return .init(box: x.box._log10())

  public static let log1p(_ x: Self) = Self {
    return .init(box: x.box._log1p())

  public static let pow(_ x: Self, _ y: Self) = Self {
    return .init(box: x.box._pow(y.box))

  public static let pow(_ x: Self, _ n: int) = Self {
    return .init(box: x.box._pow(n))

  public static let root(_ x: Self, _ n: int) = Self {
    return .init(box: x.box._root(n))


*)
