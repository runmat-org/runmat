use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MathInferenceRule {
    AngleConversion(AngleConversionInferenceRule),
    Atan2,
    Bitwise(BitwiseInferenceRule),
    ComplexConstruction,
    Discrete(DiscreteInferenceRule),
    ErrorFunction(ErrorFunctionInferenceRule),
    GammaFunction(GammaFunctionInferenceRule),
    Heaviside,
    IntegerDivide,
    Hypot,
    MagnitudePhaseSign(MagnitudePhaseSignKind),
    Exponential(ExponentialKind),
    Logarithm(LogarithmKind),
    LogicalReduction(LogicalReductionKind),
    Root(RootKind),
    NumericLimit(NumericLimitRule),
    NumericConversion(runmat_types::NumericClass),
    NumericConversionWithLike(runmat_types::NumericClass),
    NumericComponent(NumericComponentRule),
    Rounding(RoundingFunction),
    Remainder(RemainderFunction),
    Round,
    Trigonometric(TrigonometricFunction),
    Hyperbolic(HyperbolicFunction),
    PiScaledTrigonometric(PiScaledTrigonometricFunction),
    PowerOfTwo(PowerOfTwoInferenceRule),
    DegreeTrigonometric(DegreeTrigonometricFunction),
    InverseTrigonometric(InverseTrigonometricFunction),
    InverseHyperbolic(InverseHyperbolicFunction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MagnitudePhaseSignKind {
    Magnitude,
    Phase,
    Sign,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PowerOfTwoInferenceRule {
    NextExponent,
    Power,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ExponentialKind {
    Natural,
    MinusOne,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AngleConversionInferenceRule {
    DegreesToRadians,
    RadiansToDegrees,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DiscreteInferenceRule {
    Binary(BinaryNumberTheoryRule),
    Factor,
    Factorial,
    IsPrime,
    Primes,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinaryNumberTheoryRule {
    Gcd,
    Lcm,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ErrorFunctionInferenceRule {
    Erf,
    InverseComplementary,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GammaFunctionInferenceRule {
    Gamma,
    LogGamma,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BitwiseInferenceRule {
    Binary(BinaryBitwiseOperator),
    Complement,
    Get,
    Set,
    Shift,
    SwapBytes,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinaryBitwiseOperator {
    And,
    Or,
    Xor,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalReductionKind {
    All,
    Any,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RoundingFunction {
    Ceil,
    Fix,
    Floor,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RemainderFunction {
    Modulus,
    Remainder,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TrigonometricFunction {
    Sin,
    Cos,
    Tan,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HyperbolicFunction {
    Sine,
    Cosine,
    Tangent,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PiScaledTrigonometricFunction {
    Sin,
    Cos,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DegreeTrigonometricFunction {
    Sin,
    Cos,
    Tan,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InverseTrigonometricFunction {
    Sine,
    Cosine,
    Tangent,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum InverseHyperbolicFunction {
    Cosine,
    Sine,
    Tangent,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogarithmKind {
    Natural,
    OnePlus,
    Binary,
    Common,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RootKind {
    Principal,
    RealOnly,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NumericLimitRule {
    Integer(IntegerLimitKind),
    Floating(FloatingLimitKind),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum IntegerLimitKind {
    Minimum,
    Maximum,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FloatingLimitKind {
    SmallestNormal,
    LargestFinite,
    LargestConsecutiveInteger,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NumericComponentRule {
    Conjugate,
    ImaginaryPart,
    RealPart,
}
