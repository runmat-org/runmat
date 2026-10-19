use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinaryArithmeticInferenceRule {
    Add,
    Subtract,
    Multiply,
    RightDivide,
    LeftDivide,
    Power,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MatrixArithmeticInferenceRule {
    Multiply,
    LeftDivide,
    RightDivide,
    Power,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RemainderFunction {
    Modulus,
    Remainder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LogicalReductionKind {
    All,
    Any,
}
