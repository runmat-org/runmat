use serde::{Deserialize, Serialize};

use super::{MethodName, StaticMethodName};

/// Closed set of standard method identities used for operator overload
/// dispatch. `Xor` is reached through the `xor` function rather than dedicated
/// syntax, but participates in the same overload protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorOverloadMethod {
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
    ElementwiseAnd,
    ElementwiseOr,
    Xor,
    Transpose,
    ConjugateTranspose,
}

impl OperatorOverloadMethod {
    pub const ALL: [Self; 24] = [
        Self::UnaryPlus,
        Self::UnaryMinus,
        Self::Not,
        Self::Add,
        Self::Subtract,
        Self::MatrixMultiply,
        Self::ElementwiseMultiply,
        Self::MatrixPower,
        Self::ElementwisePower,
        Self::Mldivide,
        Self::Mrdivide,
        Self::ElementwiseDivide,
        Self::ElementwiseLeftDivide,
        Self::Equal,
        Self::NotEqual,
        Self::Less,
        Self::LessEqual,
        Self::Greater,
        Self::GreaterEqual,
        Self::ElementwiseAnd,
        Self::ElementwiseOr,
        Self::Xor,
        Self::Transpose,
        Self::ConjugateTranspose,
    ];

    pub const fn name(self) -> StaticMethodName {
        match self {
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
            Self::Xor => StaticMethodName::new("xor"),
            Self::Transpose => StaticMethodName::new("transpose"),
            Self::ConjugateTranspose => StaticMethodName::new("ctranspose"),
        }
    }

    pub fn from_name(name: &MethodName) -> Option<Self> {
        Self::ALL.into_iter().find(|method| method.name().is(name))
    }
}

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
        match self.overload_method() {
            Some(method) => Some(method.name()),
            None => None,
        }
    }

    pub const fn overload_method(self) -> Option<OperatorOverloadMethod> {
        Some(match self {
            Self::UnaryPlus => OperatorOverloadMethod::UnaryPlus,
            Self::UnaryMinus => OperatorOverloadMethod::UnaryMinus,
            Self::Not => OperatorOverloadMethod::Not,
            Self::Add => OperatorOverloadMethod::Add,
            Self::Subtract => OperatorOverloadMethod::Subtract,
            Self::MatrixMultiply => OperatorOverloadMethod::MatrixMultiply,
            Self::ElementwiseMultiply => OperatorOverloadMethod::ElementwiseMultiply,
            Self::MatrixPower => OperatorOverloadMethod::MatrixPower,
            Self::ElementwisePower => OperatorOverloadMethod::ElementwisePower,
            Self::Mldivide => OperatorOverloadMethod::Mldivide,
            Self::Mrdivide => OperatorOverloadMethod::Mrdivide,
            Self::ElementwiseDivide => OperatorOverloadMethod::ElementwiseDivide,
            Self::ElementwiseLeftDivide => OperatorOverloadMethod::ElementwiseLeftDivide,
            Self::Equal => OperatorOverloadMethod::Equal,
            Self::NotEqual => OperatorOverloadMethod::NotEqual,
            Self::Less => OperatorOverloadMethod::Less,
            Self::LessEqual => OperatorOverloadMethod::LessEqual,
            Self::Greater => OperatorOverloadMethod::Greater,
            Self::GreaterEqual => OperatorOverloadMethod::GreaterEqual,
            Self::ElementwiseAnd => OperatorOverloadMethod::ElementwiseAnd,
            Self::ElementwiseOr => OperatorOverloadMethod::ElementwiseOr,
            Self::Transpose => OperatorOverloadMethod::Transpose,
            Self::ConjugateTranspose => OperatorOverloadMethod::ConjugateTranspose,
            Self::ShortCircuitAnd | Self::ShortCircuitOr => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overload_method_names_round_trip_through_typed_identity() {
        for method in OperatorOverloadMethod::ALL {
            assert_eq!(
                OperatorOverloadMethod::from_name(&method.name().owned()),
                Some(method)
            );
        }
        assert_eq!(
            OperatorOverloadMethod::from_name(&MethodName::from("notAnOperator")),
            None
        );
    }
}
