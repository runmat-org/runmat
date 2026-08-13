use serde::{Deserialize, Serialize};

/// Canonical language operator identity shared by HIR, MIR, inference, and
/// executable consumers. This describes syntax/semantics, not an opcode or a
/// runtime implementation binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperatorKind {
    UnaryPlus,
    UnaryMinus,
    Not,
    Add,
    Subtract,
    MatrixMultiply,
    ElementwiseMultiply,
    MatrixPower,
    ElementwisePower,
    Mldivide,
    Mrdivide,
    ElementwiseDivide,
    ElementwiseLeftDivide,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    ShortCircuitAnd,
    ShortCircuitOr,
    ElementwiseAnd,
    ElementwiseOr,
    Transpose,
    ConjugateTranspose,
}
