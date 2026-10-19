//! Represented real scalar/dense arithmetic shared by host execution and GPU
//! scalar receipts. Result class alone is not a computation-precision proof.
use crate::{NumericClass, NumericDomain, OperatorKind, StorageFact, ValueFact};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RealFloatingRecipe {
    operator: OperatorKind,
    computation_class: NumericClass,
    result_class: NumericClass,
}

impl RealFloatingRecipe {
    /// Only operations established by the real host arithmetic owners belong
    /// here. In particular this does not admit power, matrix contractions,
    /// complex, integer, sparse or opaque computations.
    #[inline]
    pub fn for_classes(
        operator: OperatorKind,
        left: NumericClass,
        right: NumericClass,
    ) -> Option<Self> {
        if !matches!(
            operator,
            OperatorKind::Add
                | OperatorKind::Subtract
                | OperatorKind::ElementwiseMultiply
                | OperatorKind::ElementwiseDivide
                | OperatorKind::ElementwiseLeftDivide
        ) {
            return None;
        }
        let result_class = super::floating_binary_result_class(left, right)?;
        let computation_class = if left == NumericClass::Double || right == NumericClass::Double {
            NumericClass::Double
        } else {
            NumericClass::Single
        };
        Some(Self {
            operator,
            computation_class,
            result_class,
        })
    }

    #[inline]
    pub fn for_facts(operator: OperatorKind, left: &ValueFact, right: &ValueFact) -> Option<Self> {
        if ![left, right]
            .iter()
            .all(|fact| matches!(fact.storage, StorageFact::Scalar | StorageFact::Dense))
        {
            return None;
        }
        let operator = if operator == OperatorKind::MatrixMultiply {
            if left.shape.element_count() != Some(1) && right.shape.element_count() != Some(1) {
                return None;
            }
            OperatorKind::ElementwiseMultiply
        } else {
            operator
        };
        let left = left.numeric()?;
        let right = right.numeric()?;
        if left.domain != NumericDomain::Real || right.domain != NumericDomain::Real {
            return None;
        }
        Self::for_classes(operator, left.class, right.class)
    }

    #[inline]
    pub const fn operator(self) -> OperatorKind {
        self.operator
    }
    #[inline]
    pub const fn computation_class(self) -> NumericClass {
        self.computation_class
    }
    #[inline]
    pub const fn result_class(self) -> NumericClass {
        self.result_class
    }

    /// Operands are represented host values, widened losslessly to f64. A
    /// Single intermediate is rounded before any following recipe consumes it.
    #[inline]
    pub fn evaluate(self, left: f64, right: f64) -> Option<f64> {
        let value = match self.computation_class {
            NumericClass::Single => {
                let (left, right) = (left as f32, right as f32);
                f64::from(match self.operator {
                    OperatorKind::Add => left + right,
                    OperatorKind::Subtract => left - right,
                    OperatorKind::ElementwiseMultiply => left * right,
                    OperatorKind::ElementwiseDivide => left / right,
                    OperatorKind::ElementwiseLeftDivide => right / left,
                    _ => return None,
                })
            }
            NumericClass::Double => match self.operator {
                OperatorKind::Add => left + right,
                OperatorKind::Subtract => left - right,
                OperatorKind::ElementwiseMultiply => left * right,
                OperatorKind::ElementwiseDivide => left / right,
                OperatorKind::ElementwiseLeftDivide => right / left,
                _ => return None,
            },
            _ => return None,
        };
        match self.result_class {
            NumericClass::Single => Some(f64::from(value as f32)),
            NumericClass::Double => Some(value),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_product_retains_double_then_single_and_intermediate_rounding() {
        let single = 1.0 + f64::from(f32::EPSILON);
        let double = 1.0 + f64::from(f32::EPSILON) / 2.0;
        for (left, right) in [
            (NumericClass::Single, NumericClass::Double),
            (NumericClass::Double, NumericClass::Single),
        ] {
            let recipe =
                RealFloatingRecipe::for_classes(OperatorKind::ElementwiseMultiply, left, right)
                    .unwrap();
            assert_eq!(recipe.computation_class(), NumericClass::Double);
            assert_eq!(recipe.result_class(), NumericClass::Single);
            let (a, b) = if left == NumericClass::Single {
                (single, double)
            } else {
                (double, single)
            };
            let actual = recipe.evaluate(a, b).unwrap();
            assert_eq!(actual, f64::from((single * double) as f32));
            assert_ne!(actual, f64::from((single as f32) * (double as f32)));
            let next = RealFloatingRecipe::for_classes(
                OperatorKind::Subtract,
                NumericClass::Single,
                NumericClass::Double,
            )
            .unwrap();
            assert_eq!(
                next.evaluate(actual, 1.0).unwrap(),
                f64::from((f64::from((single * double) as f32) - 1.0) as f32)
            );
        }
        assert!(RealFloatingRecipe::for_classes(
            OperatorKind::ElementwisePower,
            NumericClass::Single,
            NumericClass::Double
        )
        .is_none());
    }
}
