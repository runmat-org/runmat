use serde::Serialize;

mod arithmetic;
mod elementary;
mod specialized;

pub use arithmetic::*;
pub use elementary::*;
pub use specialized::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MathInferenceRule {
    AngleConversion(AngleConversionInferenceRule),
    Atan2,
    BinaryArithmetic(BinaryArithmeticInferenceRule),
    MatrixArithmetic(MatrixArithmeticInferenceRule),
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
    Rescale,
    Typecast,
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
