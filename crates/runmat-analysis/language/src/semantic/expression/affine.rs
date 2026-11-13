//! Affine source expressions cannot be admitted solely by equal SI dimensions.
//! Kelvin literals without an explicit origin remain contextual; Celsius literals
//! declare an absolute origin, and differences of absolute temperatures are linear.
use super::{CheckedExpression, CheckedExpressionKind};
use runmat_design_syntax::{FeaBinaryOperator, FeaUnaryOperator};
use runmat_types::{
    physical::{QuantityKind, UnitError},
    PhysicalDimension,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Affinity {
    Contextual,
    AbsoluteTemperature,
    TemperatureDifference,
}

impl CheckedExpression {
    pub(crate) fn quantity_kind_hint(&self) -> Result<Option<QuantityKind>, UnitError> {
        Ok(match affinity(self)? {
            Affinity::Contextual => self
                .value_type
                .quantity_descriptor()
                .and_then(|spec| spec.kind()),
            Affinity::AbsoluteTemperature => Some(QuantityKind::Temperature),
            Affinity::TemperatureDifference => Some(QuantityKind::TemperatureDifference),
        })
    }
    pub(crate) fn validate_quantity_kind(&self, kind: QuantityKind) -> Result<(), UnitError> {
        if let Some(spec) = self.value_type.quantity_descriptor() {
            if spec.angle_power() != kind.angle_power() {
                return Err(UnitError::AngleMismatch);
            }
        }
        if self
            .quantity_kind_hint()?
            .is_some_and(|actual| actual != kind)
        {
            return Err(UnitError::KindMismatch);
        }
        match affinity(self)? {
            Affinity::AbsoluteTemperature if kind != QuantityKind::Temperature => {
                Err(UnitError::InvalidAffineUnit)
            }
            Affinity::TemperatureDifference if kind == QuantityKind::Temperature => {
                Err(UnitError::InvalidAffineUnit)
            }
            _ => Ok(()),
        }
    }
}
fn affinity(value: &CheckedExpression) -> Result<Affinity, UnitError> {
    use Affinity as A;
    Ok(match &value.kind {
        CheckedExpressionKind::Quantity {
            unit,
            source_unit_symbol,
            ..
        } if unit.dimension() == PhysicalDimension::TEMPERATURE => {
            if unit.offset() != 0.0 {
                A::AbsoluteTemperature
            } else if source_unit_symbol.trim() == "delta_degC" {
                A::TemperatureDifference
            } else {
                A::Contextual
            }
        }
        CheckedExpressionKind::CadParameterValue(value) => match value.measure.kind() {
            QuantityKind::Temperature => A::AbsoluteTemperature,
            QuantityKind::TemperatureDifference => A::TemperatureDifference,
            _ => A::Contextual,
        },
        CheckedExpressionKind::Unary { operator, operand } => {
            let operand = affinity(operand)?;
            if *operator == FeaUnaryOperator::Negative && operand == A::AbsoluteTemperature {
                return Err(UnitError::InvalidAffineUnit);
            }
            operand
        }
        CheckedExpressionKind::Binary {
            left,
            operator,
            right,
        } => {
            let (left, right) = (affinity(left)?, affinity(right)?);
            match operator {
                FeaBinaryOperator::Add => match (left, right) {
                    (A::AbsoluteTemperature, A::AbsoluteTemperature) => {
                        return Err(UnitError::InvalidAffineUnit)
                    }
                    (A::AbsoluteTemperature, _) | (_, A::AbsoluteTemperature) => {
                        A::AbsoluteTemperature
                    }
                    (A::TemperatureDifference, A::TemperatureDifference) => {
                        A::TemperatureDifference
                    }
                    _ => A::Contextual,
                },
                FeaBinaryOperator::Subtract => match (left, right) {
                    (A::AbsoluteTemperature, A::AbsoluteTemperature) => A::TemperatureDifference,
                    (A::TemperatureDifference, A::AbsoluteTemperature) => {
                        return Err(UnitError::InvalidAffineUnit)
                    }
                    (A::AbsoluteTemperature, A::TemperatureDifference) => A::AbsoluteTemperature,
                    (A::TemperatureDifference, A::TemperatureDifference) => {
                        A::TemperatureDifference
                    }
                    _ => A::Contextual,
                },
                _ => {
                    if left == A::AbsoluteTemperature || right == A::AbsoluteTemperature {
                        return Err(UnitError::InvalidAffineUnit);
                    }
                    A::Contextual
                }
            }
        }
        CheckedExpressionKind::List(values) => {
            let mut kind = A::Contextual;
            for value in values {
                let item = affinity(value)?;
                if item != A::Contextual {
                    if kind != A::Contextual && kind != item {
                        return Err(UnitError::InvalidAffineUnit);
                    }
                    kind = item;
                }
            }
            kind
        }
        _ => match value
            .value_type
            .quantity_descriptor()
            .and_then(|spec| spec.kind())
        {
            Some(QuantityKind::Temperature) => A::AbsoluteTemperature,
            Some(QuantityKind::TemperatureDifference) => A::TemperatureDifference,
            _ => A::Contextual,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeaType;
    use runmat_types::{physical::parse_unit_expression, Span};
    fn quantity(symbol: &str, value: &str) -> CheckedExpression {
        CheckedExpression {
            kind: CheckedExpressionKind::Quantity {
                unit: parse_unit_expression(symbol).unwrap(),
                source_unit_symbol: symbol.into(),
                value: Box::new(CheckedExpression {
                    kind: CheckedExpressionKind::Number(value.into()),
                    value_type: FeaType::Real,
                    span: Span { start: 0, end: 1 },
                }),
            },
            value_type: FeaType::Quantity((PhysicalDimension::TEMPERATURE).into()),
            span: Span { start: 0, end: 1 },
        }
    }
    fn binary(
        left: CheckedExpression,
        operator: FeaBinaryOperator,
        right: CheckedExpression,
    ) -> CheckedExpression {
        CheckedExpression {
            kind: CheckedExpressionKind::Binary {
                left: Box::new(left),
                operator,
                right: Box::new(right),
            },
            value_type: FeaType::Quantity((PhysicalDimension::TEMPERATURE).into()),
            span: Span { start: 0, end: 1 },
        }
    }
    #[test]
    fn celsius_origin_cannot_be_reinterpreted_as_a_temperature_jump() {
        let absolute = quantity("degC", "10");
        assert!(absolute
            .validate_quantity_kind(QuantityKind::Temperature)
            .is_ok());
        assert!(absolute
            .validate_quantity_kind(QuantityKind::TemperatureDifference)
            .is_err());
        assert!(quantity("K", "10")
            .validate_quantity_kind(QuantityKind::TemperatureDifference)
            .is_ok());
        assert!(quantity("delta_degC", "10")
            .validate_quantity_kind(QuantityKind::TemperatureDifference)
            .is_ok());
        assert!(quantity("delta_degC", "10")
            .validate_quantity_kind(QuantityKind::Temperature)
            .is_err());
    }
    #[test]
    fn difference_of_two_absolute_temperatures_is_admitted_as_linear() {
        let difference = binary(
            quantity("degC", "30"),
            FeaBinaryOperator::Subtract,
            quantity("degC", "20"),
        );
        assert!(difference
            .validate_quantity_kind(QuantityKind::TemperatureDifference)
            .is_ok());
        assert!(difference
            .validate_quantity_kind(QuantityKind::Temperature)
            .is_err());
    }
    #[test]
    fn summing_two_absolute_origins_is_rejected() {
        let sum = binary(
            quantity("degC", "30"),
            FeaBinaryOperator::Add,
            quantity("degC", "20"),
        );
        assert!(sum
            .validate_quantity_kind(QuantityKind::Temperature)
            .is_err());
    }
    #[test]
    fn translating_an_absolute_temperature_by_a_difference_preserves_its_kind() {
        let shifted = binary(
            quantity("degC", "20"),
            FeaBinaryOperator::Add,
            quantity("delta_degC", "10"),
        );
        assert!(shifted
            .validate_quantity_kind(QuantityKind::Temperature)
            .is_ok());
        assert!(shifted
            .validate_quantity_kind(QuantityKind::TemperatureDifference)
            .is_err());
    }
}
