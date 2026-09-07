use serde::Serialize;

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
