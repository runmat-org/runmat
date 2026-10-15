use serde::{Deserialize, Serialize};

use crate::{NumericClass, OperatorKind};

/// Closed scalar operation identity for a proved pure fusion slice.
///
/// This is not a second builtin registry: the selected canonical binding or
/// operator resolves to one of these operations before a physical program is
/// constructed. Numerical-domain admission remains with the operation owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarOperation {
    Operator(OperatorKind),
    /// Real Single+Double addition evaluates in Double, then rounds the sum
    /// once to Single. Operand order is retained by the Apply node.
    MixedRealAddToSingle,
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
            Self::MixedRealAddToSingle => Some(2),
            Self::Cast(NumericClass::Single | NumericClass::Double) => Some(1),
            Self::Cast(_) => None,
        }
    }
}
