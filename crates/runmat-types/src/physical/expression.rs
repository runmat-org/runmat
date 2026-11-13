//! Shared checked source-expression algebra. Domain ASTs retain their syntax,
//! while SI, nominal-angle and affine semantic rules have one implementation.
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantityBinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum QuantityArithmeticError {
    #[error("quantity dimensions or nominal units differ")]
    ResultTypeMismatch,
    #[error("incompatible semantic quantity kinds")]
    WrongQuantityKind,
    #[error("quantity dimension exponent overflow")]
    UnsupportedTypeCombination,
    #[error("division by zero")]
    DivisionByZero,
    #[error("non-finite quantity result")]
    NonFiniteResult,
}
pub fn evaluate_quantity_binary(
    operator: QuantityBinaryOperator,
    left: super::DimensionedScalar,
    right: super::DimensionedScalar,
) -> Result<super::DimensionedScalar, QuantityArithmeticError> {
    use crate::{
        physical::{DimensionedScalar, QuantityKind as K},
        PhysicalDimension as D,
    };
    use QuantityArithmeticError as E;
    use QuantityBinaryOperator as Op;
    let (a, b) = (left.semantic_kind(), right.semantic_kind());
    let (dimension, kind, value) = match operator {
        Op::Add | Op::Subtract => {
            if left.dimension() != right.dimension() || left.angle_power() != right.angle_power() {
                return Err(E::ResultTypeMismatch);
            }
            let kind = match (operator, a, b) {
                (Op::Add, Some(K::Temperature), Some(K::Temperature))
                | (Op::Subtract, Some(K::TemperatureDifference), Some(K::Temperature)) => {
                    return Err(E::WrongQuantityKind)
                }
                (Op::Subtract, Some(K::Temperature), Some(K::Temperature)) => {
                    Some(K::TemperatureDifference)
                }
                (Op::Add, Some(K::Temperature), Some(K::TemperatureDifference))
                | (Op::Add, Some(K::TemperatureDifference), Some(K::Temperature))
                | (Op::Subtract, Some(K::Temperature), Some(K::TemperatureDifference)) => {
                    Some(K::Temperature)
                }
                (_, Some(a), Some(b)) if a == b => Some(a),
                (_, Some(_), Some(_)) => return Err(E::WrongQuantityKind),
                (Op::Add, Some(K::Temperature), None) | (Op::Add, None, Some(K::Temperature)) => {
                    Some(K::Temperature)
                }
                (Op::Subtract, Some(K::Temperature), None) => None,
                (Op::Subtract, None, Some(K::Temperature)) => Some(K::TemperatureDifference),
                (_, Some(K::TemperatureDifference), None)
                | (_, None, Some(K::TemperatureDifference)) => None,
                (_, Some(a), None) | (_, None, Some(a)) => Some(a),
                _ => None,
            };
            (
                left.descriptor().without_semantic_kind(),
                kind,
                if operator == Op::Add {
                    left.canonical_value() + right.canonical_value()
                } else {
                    left.canonical_value() - right.canonical_value()
                },
            )
        }
        Op::Multiply | Op::Divide => {
            if a == Some(K::Temperature) || b == Some(K::Temperature) {
                return Err(E::WrongQuantityKind);
            }
            if operator == Op::Divide && right.canonical_value() == 0.0 {
                return Err(E::DivisionByZero);
            }
            let dimension = if operator == Op::Multiply {
                left.descriptor().multiply(right.descriptor())
            } else {
                left.descriptor().divide(right.descriptor())
            }
            .ok_or(E::UnsupportedTypeCombination)?;
            let kind = if right.dimension() == D::DIMENSIONLESS && right.angle_power() == 0 {
                a
            } else if operator == Op::Multiply
                && left.dimension() == D::DIMENSIONLESS
                && left.angle_power() == 0
            {
                b
            } else {
                None
            };
            (
                dimension,
                kind,
                if operator == Op::Multiply {
                    left.canonical_value() * right.canonical_value()
                } else {
                    left.canonical_value() / right.canonical_value()
                },
            )
        }
    };
    DimensionedScalar::try_from_descriptor(
        dimension
            .with_semantic_kind(kind)
            .map_err(|_| E::WrongQuantityKind)?,
        value,
    )
    .map_err(|_| E::NonFiniteResult)
}
