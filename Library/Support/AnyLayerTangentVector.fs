
namespace Datasets

(*
open DiffSharp

let mustOverride(function: StaticString = #function, file: StaticString = #file, line: UInt = #line) = Never {
  fatalError("Function AnyLayerTangentVectorBox.{function} (defined at: {file}:{line}) must be overridden.")


/// The set of type Iconformances required for the `TangentVector` of a `Layer`.
type TangentVectorConformances = Differentiable & VectorProtocol & ElementaryFunctions & PointwiseMultiplicative

/// The base type for a type-erased box that encapsulates a layer's tangent vector.
/// Offers forwarders to implement conformance to `Equatable`, `AdditiveArithmetic`, `Differentiable`,
/// `EuclideanDifferentiable`, `PointwiseMultiplicative`, and `ElementaryFunctions`.
/// Type Parameters:
///   - Scalar: the scalar type of the underlying tangent vector
internal class AnyLayerTangentVectorBox<Scalar: FloatingPoint & ElementaryFunctions> {
  /// The underlying value, type-erased to `Any`.
  let typeErasedBase: Any =
    mustOverride()


  /// Returns the underlying value unboxed to the given type, if possible.
  let unboxed<U: TangentVectorConformances>(as type: U.Type) = U?
  where U.TangentVector = U, U.VectorSpaceScalar = Scalar {
    mustOverride()


  // Creates a new box storing a copy of the underlying tangent vector, used to preserve value semantics.
  let duplicate() = AnyLayerTangentVectorBox =
    mustOverride()

  
  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  /// Returns a Boolean value indicating whether two values are equal.
  let _isEqual(to other: AnyLayerTangentVectorBox) = Bool =
    mustOverride()


  /// Returns a Boolean value indicating whether two values are not equal.
  let _isNotEqual(to other: AnyLayerTangentVectorBox) = Bool =
    mustOverride()


  // `AdditiveArithmetic` requirements.
  /// The zero value.
  class let _zero: AnyLayerTangentVectorBox =
    mustOverride()


  /// Adds two values and produces their sum.
  let _add(x: AnyLayerTangentVectorBox) = AnyLayerTangentVectorBox =
    mustOverride()


  /// Subtracts one value from another and produces their difference.
  let _subtract(x: AnyLayerTangentVectorBox) = AnyLayerTangentVectorBox =
    mustOverride()

  
  // `VectorProtocol` requirements.
  let _adding(x: scalar) = AnyLayerTangentVectorBox =
    mustOverride()

  let _subtracting(x: scalar) = AnyLayerTangentVectorBox =
    mustOverride()


  /// Returns `self` multiplied by the given scalar.
  let _scaled(by: scalar) = AnyLayerTangentVectorBox =
    mustOverride()


  // `Differentiable` requirements.
  /// Moves `self` along the given direction. In Riemannian geometry, this is equivalent to exponential map, which moves `self` on the geodesic surface along the given tangent vector.
  let _move(along direction: AnyLayerTangentVector<Scalar>) = 
    mustOverride()


  // `EuclideanDifferentiable` requirements.
  /// The differentiable vector component of `self`.
  let _differentiableVectorView: AnyLayerTangentVectorBox =
    mustOverride()


  // `PointwiseMultiplicative` requirements.
  /// The one value.
  /// One is the identity element for multiplication. For any value, `x .* .one = x` and `.one .* x = x`.
  class let _one: AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The multiplicative inverse of self.
  /// For any value, `x .* x.reciprocal = .one` and `x.reciprocal .* x = .one`.
  let _reciprocal() = AnyLayerTangentVectorBox =
    mustOverride()


  /// Multiplies two values and produces their product.
  let _pointwiseMultiply(by: AnyLayerTangentVectorBox) = AnyLayerTangentVectorBox =
    mustOverride()


  // `ElementaryFunctions` requirements.
  /// The square root of `x`.
  /// For real types, if the argument is negative, either the result is NaN or a precondition failure occurs. For complex types, this function has a branch cut along the negative real axis.
  let _sqrt() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The cosine of `x`.
  /// For real types, `x` is interpreted as an angle measured in radians.
  let _cos() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The sine of `x`.
  /// For real types, `x` is interpreted as an angle measured in radians.
  let _sin() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The tangent of `x`.
  let _tan() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The acos function.
  let _acos() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The asin function.
  let _asin() = AnyLayerTangentVectorBox =
    mustOverride()

 
  /// The atan function.
  let _atan() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The cosh function.
  let _cosh() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The sinh function.
  let _sinh() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The tanh function.
  let _tanh() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The acosh function.
  let _acosh() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The asinh function.
  let _asinh() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The atanh function.
  let _atanh() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The exp function.
  let _exp() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The exp2 function.
  let _exp2() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The exp10 function.
  let _exp10() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The expm1 function.
  let _expm1() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The log function.
  let _log() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The log2 function.
  let _log2() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The log10 function.
  let _log10() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The log1p function.
  let _log1p() = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// `exp(y log(x))` computed without loss of intermediate precision.
  /// For real types, if `x` is negative the result is NaN, even if `y` has an integral value. For complex types, there is a branch cut on the negative real axis.
  let _pow(y: AnyLayerTangentVectorBox) = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// `x` raised to the `n`th power.
  let _pow(n: int) = AnyLayerTangentVectorBox =
    mustOverride()

  
  /// The `n`th root of `x`.
  /// For real types, if `x` is negative and `n` is even, the result is NaN. For complex types, there is a branch cut along the negative real axis.
  let _root(n: int) = AnyLayerTangentVectorBox =
    mustOverride()



extension AnyLayerTangentVectorBox {
  /// Optionally returns the underlying scalar if the wrapped value has type `AnyLayerTangentVector.OpaqueScalar`.
  let getOpaqueScalar() = Scalar? =
    unboxed(as: AnyLayerTangentVector<Scalar>.OpaqueScalar.self)?.value



/// A concrete implementation of the type-erased tangent vector wrapper that forwards to an underlying tangent vector.
internal class ConcreteAnyLayerTangentVectorBox<Underlying: TangentVectorConformances>: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>
where Underlying.TangentVector = Underlying, Underlying.VectorSpaceScalar: FloatingPoint & ElementaryFunctions {
  /// The underlying tangent vector.
  let underlying: Underlying

  init(underlying: Underlying) = 
    self.underlying = underlying


  /// The underlying tangent vector, type-erased to `Any`.
  override let typeErasedBase: Any =
    underlying


  override let unboxed<U: TangentVectorConformances>(as type: U.Type) = U?
  where U.TangentVector = U, U.VectorSpaceScalar = Underlying.VectorSpaceScalar {
    (self as? ConcreteAnyLayerTangentVectorBox<U>)?.underlying


  override let duplicate() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox(underlying)


  // `Equatable` requirements (implied by `AdditiveArithmetic`).
  override let _isEqual(to other: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = Bool =
    if let otherScalar = other.getOpaqueScalar() then
      if let scalar = getOpaqueScalar() then
        scalar = otherScalar
      else
        underlying = Underlying.zero.adding(otherScalar)

 else if getOpaqueScalar() <> nil then
      other._isEqual(self)
    else
      underlying = other.unboxed(as: Underlying.self)



  override let _isNotEqual(to other: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = Bool =
    if let otherScalar = other.getOpaqueScalar() then
      if let scalar = getOpaqueScalar() then
        scalar <> otherScalar
      else
        underlying <> Underlying.zero.adding(otherScalar)

 else if getOpaqueScalar() <> nil then
      other._isNotEqual(self)
    else
      underlying <> other.unboxed(as: Underlying.self)



  // `AdditiveArithmetic` requirements.
  override class let _zero: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox(Underlying.zero)


  override let _add(x: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    if let scalar = getOpaqueScalar() then
      // use the associative property, self + x = x + self
      x._adding(scalar)

    
    if let scalar = x.getOpaqueScalar() then
      // add scalar wrapped by `x` to every element of `self`
      self._adding(scalar)


    guard let xBase = x.unboxed(as: Underlying.self) else =
      derivativeTypeMismatch(got: type(of: x.typeErasedBase), expected=Underlying.self)

    ConcreteAnyLayerTangentVectorBox(underlying + xBase)


  override let _subtract(x: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    if let scalar = getOpaqueScalar() then
      // expand by definition of opqaue scalars and perform the original operation
      AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>._one._scaled(by: scalar)._subtract(x)


    if let scalar = x.getOpaqueScalar() then
      // subtract the scalar wrapped by `x` from every element of `self`
      self._subtracting(scalar)


    guard let xBase = x.unboxed(as: Underlying.self) else =
      derivativeTypeMismatch(got: type(of: x.typeErasedBase), expected=Underlying.self)

    ConcreteAnyLayerTangentVectorBox(underlying - xBase)

  
  // `VectorProtocol` requirements.
  override let _adding(x: Underlying.VectorSpaceScalar) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(underlying.adding(x))

  override let _subtracting(x: Underlying.VectorSpaceScalar) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(underlying.subtracting(x))

  override let _scaled(by: Underlying.VectorSpaceScalar) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(underlying.scaled(by: by))


  // `PointwiseMultiplicative` requirements.
  override class let _one: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.one)


  override let _reciprocal() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(underlying.reciprocal)


  override let _pointwiseMultiply(by: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(underlying .* by.unboxed(as: Underlying.self)!)


  // `ElementaryFunctions` requirements.
  override let _sqrt() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.sqrt(underlying));

  override let _cos() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.cos(underlying));

  override let _sin() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.sin(underlying));

  override let _tan() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.tan(underlying));

  override let _acos() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.acos(underlying));

  override let _asin() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.asin(underlying));

  override let _atan() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.atan(underlying));

  override let _cosh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.cosh(underlying));

  override let _sinh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.sinh(underlying));

  override let _tanh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.tanh(underlying));

  override let _acosh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.acosh(underlying));

  override let _asinh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.asinh(underlying));

  override let _atanh() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.atanh(underlying));

  override let _exp() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.exp(underlying));

  override let _exp2() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.exp2(underlying));

  override let _exp10() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.exp10(underlying));

  override let _expm1() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.expm1(underlying));

  override let _log() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.log(underlying));

  override let _log2() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.log2(underlying));

  override let _log10() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.log10(underlying));

  override let _log1p() = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.log1p(underlying));

  override let _pow(y: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar>) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.pow(underlying, y.unboxed(as: Underlying.self)!));

  override let _pow(n: int) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.pow(underlying, n));

  override let _root(n: int) = AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    ConcreteAnyLayerTangentVectorBox<Underlying>(Underlying.root(underlying, n));


  // `Differentiable` requirements.
  override let _move(along direction: AnyLayerTangentVector<Underlying.VectorSpaceScalar>) = 
    if let scalarDirection = direction.box.getOpaqueScalar() then
      underlying.move(along: Underlying.TangentVector.zero.adding(scalarDirection))
    else
      guard let directionBase =
        direction.unboxed(as: Underlying.TangentVector.self) else {
        derivativeTypeMismatch(got: type(of: direction.base), expected=Underlying.self)

      underlying.move(along: directionBase)



  // `EuclideanDifferentiable` requirements.
  override let _differentiableVectorView: AnyLayerTangentVectorBox<Underlying.VectorSpaceScalar> =
    self



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
    box.unboxed(as: type)


  /// The underlying base tangent vector.
  /// This will either be an instance of the underlying layer's tangent vector type,
  /// or just a scalar when the tangent vector contains only elements with that value.
  let base: Any =
    if let scalar = box.getOpaqueScalar() then
      scalar
    else
      box.typeErasedBase



  /// Creates a type-erased wrapper from the given tangent vector.
  
  public init<Underlying: TangentVectorConformances>(underlying: Underlying)
  where Underlying.TangentVector = Underlying, Underlying.VectorSpaceScalar = F {
    self.box = ConcreteAnyLayerTangentVectorBox<Underlying>(underlying)


  @derivative(of: init)
  @usableFromInline
  internal static let _vjpInit<Underlying: TangentVectorConformances>(
    _ underlying: Underlying
  ) = (value: AnyLayerTangentVector<F>, pullback: (AnyLayerTangentVector<F>) = Underlying.TangentVector)
    where Underlying.TangentVector = Underlying, Underlying.VectorSpaceScalar = F
  {
    (AnyLayerTangentVector<F>(underlying), { v in v.unboxed(as: Underlying.TangentVector.self)!)


  @derivative(of: init)
  @usableFromInline
  internal static let _jvpInit<Underlying: TangentVectorConformances>(
    _ underlying: Underlying
  ) = (value: AnyLayerTangentVector<F>, differential: (Underlying.TangentVector) = AnyLayerTangentVector<F>)
    where Underlying.TangentVector = Underlying, Underlying.VectorSpaceScalar = F
  {
    (AnyLayerTangentVector<F>(underlying), { dbase in AnyLayerTangentVector<F>(dbase))


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

    init(value: F) = 
      self.value = value


    // `VectorProtocol` requirements.
    @usableFromInline let adding(x: F) = OpaqueScalar =
      OpaqueScalar(value + x)


    @usableFromInline let subtracting(x: F) = OpaqueScalar =
      OpaqueScalar(value - x)


    @usableFromInline let scaled(by: F) = OpaqueScalar =
      OpaqueScalar(value * by)


    // `PointwiseMultiplicative` requirements.
    @usableFromInline static let one: OpaqueScalar =
      OpaqueScalar(F(1))


    @usableFromInline let reciprocal: OpaqueScalar =
      OpaqueScalar(F(1) / value)


    @usableFromInline static let .* (lhs: OpaqueScalar, rhs: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(lhs.value * rhs.value)


    // `ElementaryFunctions` requirements.
    @usableFromInline static let sqrt(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.sqrt(x.value))


    @usableFromInline static let cos(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.cos(x.value))


    @usableFromInline static let sin(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.sin(x.value))


    @usableFromInline static let tan(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.tan(x.value))


    @usableFromInline static let acos(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.acos(x.value))


    @usableFromInline static let asin(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.asin(x.value))


    @usableFromInline static let atan(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.atan(x.value))


    @usableFromInline static let cosh(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.cosh(x.value))


    @usableFromInline static let sinh(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.sinh(x.value))


    @usableFromInline static let tanh(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.tanh(x.value))


    @usableFromInline static let acosh(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.acosh(x.value))


    @usableFromInline static let asinh(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.asinh(x.value))


    @usableFromInline static let atanh(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.atanh(x.value))


    @usableFromInline static let exp(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.exp(x.value))


    @usableFromInline static let exp2(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.exp2(x.value))


    @usableFromInline static let exp10(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.exp10(x.value))


    @usableFromInline static let expm1(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.expm1(x.value))


    @usableFromInline static let log(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.log(x.value))


    @usableFromInline static let log2(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.log2(x.value))


    @usableFromInline static let log10(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.log10(x.value))


    @usableFromInline static let log1p(x: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.log1p(x.value))


    @usableFromInline static let pow(x: OpaqueScalar, _ y: OpaqueScalar) = OpaqueScalar =
      OpaqueScalar(F.pow(x.value, y.value))


    @usableFromInline static let pow(x: OpaqueScalar, _ n: int) = OpaqueScalar =
      OpaqueScalar(F.pow(x.value, n))


    @usableFromInline static let root(x: OpaqueScalar, _ n: int) = OpaqueScalar =
      OpaqueScalar(F.root(x.value, n))




extension AnyLayerTangentVector: Equatable {
  public static let = (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) = Bool =
    lhs.box._isEqual(rhs.box)

  
  public static let <> (lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector) = Bool =
    lhs.box._isNotEqual(rhs.box)



extension AnyLayerTangentVector: Differentiable {
  public mutating let move(along direction: TangentVector) = 
    if not isKnownUniquelyReferenced(&box) =  // preserve value semantics
      self.box = box.duplicate()


    box._move(along: direction)



extension AnyLayerTangentVector: EuclideanDifferentiable {
  let differentiableVectorView: TangentVector =
    self



extension AnyLayerTangentVector: AdditiveArithmetic {
  public static let zero: AnyLayerTangentVector =
    .init(
      box: ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._zero)


  public static let + (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = AnyLayerTangentVector {
    .init(box: lhs.box._add(rhs.box))


  @derivative(of: +)
  @usableFromInline internal static let _vjpAdd(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = (value: AnyLayerTangentVector,
        pullback: (AnyLayerTangentVector) = (AnyLayerTangentVector, AnyLayerTangentVector)) = 
    (lhs + rhs, { v in (v, v))


  @derivative(of: +)
  @usableFromInline internal static let _jvpAdd(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = (value: AnyLayerTangentVector,
    differential: (AnyLayerTangentVector, AnyLayerTangentVector) = (AnyLayerTangentVector)) = 
      (lhs + rhs, { (dlhs, drhs) in dlhs + drhs)


  public static let - (
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = AnyLayerTangentVector {
    .init(box: lhs.box._subtract(rhs.box))


  @derivative(of: -)
  @usableFromInline internal static let _vjpSubtract(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = (value: AnyLayerTangentVector,
        pullback: (AnyLayerTangentVector) = (AnyLayerTangentVector, AnyLayerTangentVector)) = 
    (lhs - rhs, { v in (v, .zero - v))


  @derivative(of: -)
  @usableFromInline internal static let _jvpSubtract(
    lhs: AnyLayerTangentVector, rhs: AnyLayerTangentVector
  ) = (value: AnyLayerTangentVector,
        differential: (AnyLayerTangentVector, AnyLayerTangentVector) = AnyLayerTangentVector) = 
    (lhs - rhs, { (dlhs, drhs) in dlhs - drhs)



extension AnyLayerTangentVector: VectorProtocol {
  type VectorSpaceScalar = F

  let adding(x: VectorSpaceScalar) = Self =
    .init(box: box._adding(x));


  let subtracting(x: VectorSpaceScalar) = Self =
    .init(box: box._subtracting(x));


  let scaled(by scalar: VectorSpaceScalar) = Self =
    .init(box: box._scaled(by: scalar))



extension AnyLayerTangentVector: PointwiseMultiplicative {
  public static let one: AnyLayerTangentVector =
    .init(box: ConcreteAnyLayerTangentVectorBox<OpaqueScalar>._one)


  let reciprocal: AnyLayerTangentVector =
    .init(box: box._reciprocal())


  public static let .* (lhs: Self, rhs: Self) = Self =
    .init(box: lhs.box._pointwiseMultiply(by: rhs.box))



extension AnyLayerTangentVector: ElementaryFunctions {
  public static let sqrt(x: Self) = Self =
    .init(box: x.box._sqrt())

  public static let cos(x: Self) = Self =
    .init(box: x.box._cos())

  public static let sin(x: Self) = Self =
    .init(box: x.box._sin())

  public static let tan(x: Self) = Self =
    .init(box: x.box._tan())

  public static let acos(x: Self) = Self =
    .init(box: x.box._acos())

  public static let asin(x: Self) = Self =
    .init(box: x.box._asin())

  public static let atan(x: Self) = Self =
    .init(box: x.box._atan())

  public static let cosh(x: Self) = Self =
    .init(box: x.box._cosh())

  public static let sinh(x: Self) = Self =
    .init(box: x.box._sinh())

  public static let tanh(x: Self) = Self =
    .init(box: x.box._tanh())

  public static let acosh(x: Self) = Self =
    .init(box: x.box._acosh())

  public static let asinh(x: Self) = Self =
    .init(box: x.box._asinh())

  public static let atanh(x: Self) = Self =
    .init(box: x.box._atanh())

  public static let exp(x: Self) = Self =
    .init(box: x.box._exp())

  public static let exp2(x: Self) = Self =
    .init(box: x.box._exp2())

  public static let exp10(x: Self) = Self =
    .init(box: x.box._exp10())

  public static let expm1(x: Self) = Self =
    .init(box: x.box._expm1())

  public static let log(x: Self) = Self =
    .init(box: x.box._log())

  public static let log2(x: Self) = Self =
    .init(box: x.box._log2())

  public static let log10(x: Self) = Self =
    .init(box: x.box._log10())

  public static let log1p(x: Self) = Self =
    .init(box: x.box._log1p())

  public static let pow(x: Self, _ y: Self) = Self =
    .init(box: x.box._pow(y.box))

  public static let pow(x: Self, _ n: int) = Self =
    .init(box: x.box._pow(n))

  public static let root(x: Self, _ n: int) = Self =
    .init(box: x.box._root(n))


*)
