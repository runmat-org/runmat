use serde::{Deserialize, Serialize};

use super::StaticMethodName;

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
    pub const fn overload_name(self) -> Option<StaticMethodName> {
        Some(match self {
            Self::UnaryPlus => StaticMethodName::new("uplus"),
            Self::UnaryMinus => StaticMethodName::new("uminus"),
            Self::Not => StaticMethodName::new("not"),
            Self::Add => StaticMethodName::new("plus"),
            Self::Subtract => StaticMethodName::new("minus"),
            Self::MatrixMultiply => StaticMethodName::new("mtimes"),
            Self::ElementwiseMultiply => StaticMethodName::new("times"),
            Self::MatrixPower => StaticMethodName::new("mpower"),
            Self::ElementwisePower => StaticMethodName::new("power"),
            Self::Mldivide => StaticMethodName::new("mldivide"),
            Self::Mrdivide => StaticMethodName::new("mrdivide"),
            Self::ElementwiseDivide => StaticMethodName::new("rdivide"),
            Self::ElementwiseLeftDivide => StaticMethodName::new("ldivide"),
            Self::Equal => StaticMethodName::new("eq"),
            Self::NotEqual => StaticMethodName::new("ne"),
            Self::Less => StaticMethodName::new("lt"),
            Self::LessEqual => StaticMethodName::new("le"),
            Self::Greater => StaticMethodName::new("gt"),
            Self::GreaterEqual => StaticMethodName::new("ge"),
            Self::ElementwiseAnd => StaticMethodName::new("and"),
            Self::ElementwiseOr => StaticMethodName::new("or"),
            Self::Transpose => StaticMethodName::new("transpose"),
            Self::ConjugateTranspose => StaticMethodName::new("ctranspose"),
            Self::ShortCircuitAnd | Self::ShortCircuitOr => return None,
        })
    }
}
