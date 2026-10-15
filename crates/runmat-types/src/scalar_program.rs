use serde::{Deserialize, Serialize};

use crate::{NumericClass, OperatorKind, RealFloatingRecipe};

/// Numerical instructions, not builtin names or binding authority. Domain
/// admission and original binding selection remain with canonical owners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarUnaryMath {
    Exp,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Atan,
    Asinh,
    Acosh,
    Atanh,
    Log,
    Log1p,
    Log10,
    Log2,
    Sqrt,
    Asin,
    Acos,
}

impl ScalarUnaryMath {
    /// Real-input domain of the numerical instruction, independently of any
    /// builtin's complex-result, error, or residency/publication convention.
    pub const fn real_predicate(self) -> Option<crate::RealUnaryPredicate> {
        use crate::RealUnaryPredicate as Predicate;
        match self {
            Self::Log | Self::Log10 | Self::Log2 | Self::Sqrt => Some(Predicate::Nonnegative),
            Self::Log1p => Some(Predicate::AtLeastMinusOne),
            Self::Asin | Self::Acos | Self::Atanh => Some(Predicate::ClosedUnitInterval),
            Self::Acosh => Some(Predicate::AtLeastOne),
            Self::Exp
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Sinh
            | Self::Cosh
            | Self::Tanh
            | Self::Atan
            | Self::Asinh => None,
        }
    }
}

/// Closed scalar operation identity for a proved pure fusion slice.
///
/// This is not a second builtin registry: the selected canonical binding or
/// operator resolves to one of these operations before a physical program is
/// constructed. Numerical-domain admission remains with the operation owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarOperation {
    Operator(OperatorKind),
    /// Canonical real arithmetic with explicit computation and result classes.
    /// Operand order and one result conversion belong to the original Apply.
    RealArithmetic(RealFloatingRecipe),
    UnaryMath(ScalarUnaryMath),
    Cast(NumericClass),
}

impl ScalarOperation {
    pub const fn arity(self) -> Option<usize> {
        match self {
            Self::Operator(operation) => match operation {
                OperatorKind::UnaryPlus | OperatorKind::UnaryMinus => Some(1),
                OperatorKind::Add
                | OperatorKind::Subtract
                | OperatorKind::ElementwiseMultiply
                | OperatorKind::ElementwiseDivide
                | OperatorKind::ElementwiseLeftDivide
                | OperatorKind::ElementwisePower => Some(2),
                _ => None,
            },
            Self::RealArithmetic(_) => Some(2),
            Self::UnaryMath(_) => Some(1),
            Self::Cast(NumericClass::Single | NumericClass::Double) => Some(1),
            Self::Cast(_) => None,
        }
    }
}
