use serde::Serialize;

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
pub enum RoundingFunction {
    Ceil,
    Fix,
    Floor,
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
pub enum NumericComponentRule {
    Conjugate,
    ImaginaryPart,
    RealPart,
}
