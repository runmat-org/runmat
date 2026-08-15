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

impl OperatorKind {
    /// MATLAB method/builtin identity used when an operator reaches dynamic
    /// overload dispatch. Short-circuit operators are control flow and have no
    /// callable overload edge at this layer.
    pub const fn overload_name(self) -> Option<&'static str> {
        Some(match self {
            Self::UnaryPlus => "uplus",
            Self::UnaryMinus => "uminus",
            Self::Not => "not",
            Self::Add => "plus",
            Self::Subtract => "minus",
            Self::MatrixMultiply => "mtimes",
            Self::ElementwiseMultiply => "times",
            Self::MatrixPower => "mpower",
            Self::ElementwisePower => "power",
            Self::Mldivide => "mldivide",
            Self::Mrdivide => "mrdivide",
            Self::ElementwiseDivide => "rdivide",
            Self::ElementwiseLeftDivide => "ldivide",
            Self::Equal => "eq",
            Self::NotEqual => "ne",
            Self::Less => "lt",
            Self::LessEqual => "le",
            Self::Greater => "gt",
            Self::GreaterEqual => "ge",
            Self::ElementwiseAnd => "and",
            Self::ElementwiseOr => "or",
            Self::Transpose => "transpose",
            Self::ConjugateTranspose => "ctranspose",
            Self::ShortCircuitAnd | Self::ShortCircuitOr => return None,
        })
    }
}
