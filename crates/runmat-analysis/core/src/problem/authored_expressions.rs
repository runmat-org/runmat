use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    authored_inputs::{AuthoredInputDefinition, AuthoredInputType, AuthoredInputValue},
    model_authoring::AuthoredModelInputRef,
};

/// A closed, pure numeric expression over reusable-model inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredNumericExpression {
    pub value_type: AuthoredInputType,
    pub kind: AuthoredNumericExpressionKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AuthoredNumericExpressionKind {
    Literal {
        value: f64,
    },
    Quantity {
        value: Box<AuthoredNumericExpression>,
        unit: runmat_types::physical::UnitDefinition,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        semantic_kind: Option<runmat_types::physical::QuantityKind>,
        source_unit_symbol: String,
    },
    ModelInput {
        input: AuthoredModelInputRef,
    },
    Negate {
        value: Box<AuthoredNumericExpression>,
    },
    Binary {
        operator: AuthoredNumericOperator,
        left: Box<AuthoredNumericExpression>,
        right: Box<AuthoredNumericExpression>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoredNumericOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl AuthoredNumericExpression {
    pub fn validate(
        &self,
        inputs: &[AuthoredInputDefinition],
    ) -> Result<(), AuthoredExpressionError> {
        let inputs = inputs
            .iter()
            .map(|input| (input.input_id.as_str(), input.value_type))
            .collect::<BTreeMap<_, _>>();
        self.validate_with(&inputs)
    }

    /// Explicit scalar projection at a numerical consumer. Physical consumers
    /// should use evaluate_for_quantity so known semantic kinds are checked too.
    pub fn evaluate(
        &self,
        inputs: &BTreeMap<String, AuthoredInputValue>,
    ) -> Result<f64, AuthoredExpressionError> {
        Ok(self.evaluate_dimensioned(inputs)?.canonical_value())
    }

    pub fn evaluate_for_quantity(
        &self,
        inputs: &BTreeMap<String, AuthoredInputValue>,
        expected: runmat_types::physical::QuantityKind,
    ) -> Result<f64, AuthoredExpressionError> {
        let value = self.evaluate_dimensioned(inputs)?;
        value
            .require_kind(expected)
            .map_err(|_| AuthoredExpressionError::WrongQuantityKind)?;
        Ok(value.canonical_value())
    }

    pub fn evaluate_dimensioned(
        &self,
        inputs: &BTreeMap<String, AuthoredInputValue>,
    ) -> Result<runmat_types::physical::DimensionedScalar, AuthoredExpressionError> {
        use runmat_types::{
            physical::{DimensionedScalar, QuantityKind as K},
            PhysicalDimension as D,
        };
        let value = match &self.kind {
            AuthoredNumericExpressionKind::Literal { value } => {
                DimensionedScalar::try_from_dimension(D::DIMENSIONLESS, None, *value)
                    .map_err(|_| AuthoredExpressionError::NonFiniteLiteral)?
            }
            AuthoredNumericExpressionKind::Quantity {
                value,
                unit,
                semantic_kind,
                ..
            } => {
                let source = value.evaluate_dimensioned(inputs)?;
                if source.dimension() != D::DIMENSIONLESS || source.angle_power() != 0 {
                    return Err(AuthoredExpressionError::ResultTypeMismatch);
                }
                let canonical = unit
                    .to_canonical(source.canonical_value())
                    .map_err(|_| AuthoredExpressionError::NonFiniteResult)?;
                DimensionedScalar::try_from_descriptor(
                    runmat_types::physical::DynamicDimension::from_unit(*unit)
                        .with_semantic_kind(*semantic_kind)
                        .map_err(|_| AuthoredExpressionError::WrongQuantityKind)?,
                    canonical,
                )
                .map_err(|_| AuthoredExpressionError::WrongQuantityKind)?
            }
            AuthoredNumericExpressionKind::ModelInput { input } => {
                let value = inputs
                    .get(&input.input_id)
                    .ok_or_else(|| AuthoredExpressionError::UnboundInput(input.input_id.clone()))?;
                if !value.matches(input.value_type) {
                    return Err(AuthoredExpressionError::InputTypeMismatch(
                        input.input_id.clone(),
                    ));
                }
                match value {
                    AuthoredInputValue::Quantity(value) => *value,
                    AuthoredInputValue::Integer(value) => {
                        DimensionedScalar::try_from_dimension(D::DIMENSIONLESS, None, *value as f64)
                            .map_err(|_| AuthoredExpressionError::NonFiniteResult)?
                    }
                    AuthoredInputValue::Real(value) => {
                        DimensionedScalar::try_from_dimension(D::DIMENSIONLESS, None, *value)
                            .map_err(|_| AuthoredExpressionError::NonFiniteResult)?
                    }
                    _ => {
                        return Err(AuthoredExpressionError::NonNumericInput(
                            input.input_id.clone(),
                        ))
                    }
                }
            }
            AuthoredNumericExpressionKind::Negate { value } => {
                let value = value.evaluate_dimensioned(inputs)?;
                if value.semantic_kind() == Some(K::Temperature) {
                    return Err(AuthoredExpressionError::WrongQuantityKind);
                }
                DimensionedScalar::try_from_descriptor(value.descriptor(), -value.canonical_value())
                    .map_err(|_| AuthoredExpressionError::NonFiniteResult)?
            }
            AuthoredNumericExpressionKind::Binary {
                operator,
                left,
                right,
            } => evaluate_binary_quantity(
                *operator,
                left.evaluate_dimensioned(inputs)?,
                right.evaluate_dimensioned(inputs)?,
            )?,
        };
        let expected = match self.value_type {
            AuthoredInputType::Quantity(spec) => spec,
            AuthoredInputType::Integer | AuthoredInputType::Real => D::DIMENSIONLESS.into(),
            _ => return Err(AuthoredExpressionError::NonNumericType),
        };
        if !expected.is_compatible_with(value.descriptor()) {
            return Err(AuthoredExpressionError::ResultTypeMismatch);
        }
        let spec = value
            .descriptor()
            .with_semantic_kind(value.semantic_kind().or(expected.kind()))
            .map_err(|_| AuthoredExpressionError::WrongQuantityKind)?;
        DimensionedScalar::try_from_descriptor(spec, value.canonical_value())
            .map_err(|_| AuthoredExpressionError::NonFiniteResult)
    }

    fn validate_with(
        &self,
        inputs: &BTreeMap<&str, AuthoredInputType>,
    ) -> Result<(), AuthoredExpressionError> {
        if !numeric_type(self.value_type) {
            return Err(AuthoredExpressionError::NonNumericType);
        }
        match &self.kind {
            AuthoredNumericExpressionKind::Literal { value }
                if value.is_finite()
                    && !matches!(self.value_type, AuthoredInputType::Quantity(_)) =>
            {
                Ok(())
            }
            AuthoredNumericExpressionKind::Literal { .. } => {
                Err(AuthoredExpressionError::NonFiniteLiteral)
            }
            AuthoredNumericExpressionKind::Quantity {
                value,
                unit,
                semantic_kind,
                source_unit_symbol,
            } => {
                value.validate_with(inputs)?;
                if !matches!(
                    value.value_type,
                    AuthoredInputType::Real | AuthoredInputType::Integer
                ) || !self.value_type.accepts_type(AuthoredInputType::Quantity(
                    runmat_types::physical::DynamicDimension::from_unit(*unit),
                )) || semantic_kind.is_some_and(|kind| {
                    kind.dimension() != unit.dimension() || kind.angle_power() != unit.angle_power()
                }) || (unit.offset() != 0.0
                    && *semantic_kind != Some(runmat_types::physical::QuantityKind::Temperature))
                    || source_unit_symbol.trim().is_empty()
                    || source_unit_symbol.len() > 256
                    || source_unit_symbol.chars().any(char::is_control)
                {
                    return Err(AuthoredExpressionError::WrongQuantityKind);
                }
                Ok(())
            }
            AuthoredNumericExpressionKind::ModelInput { input } => {
                if input.input_id.trim().is_empty() || input.value_type != self.value_type {
                    return Err(AuthoredExpressionError::InvalidInputReference);
                }
                match inputs.get(input.input_id.as_str()) {
                    Some(value_type) if *value_type == input.value_type => Ok(()),
                    Some(_) => Err(AuthoredExpressionError::InputTypeMismatch(
                        input.input_id.clone(),
                    )),
                    None => Err(AuthoredExpressionError::UnknownInput(
                        input.input_id.clone(),
                    )),
                }
            }
            AuthoredNumericExpressionKind::Negate { value } => {
                value.validate_with(inputs)?;
                (value.value_type == self.value_type)
                    .then_some(())
                    .ok_or(AuthoredExpressionError::ResultTypeMismatch)
            }
            AuthoredNumericExpressionKind::Binary {
                operator,
                left,
                right,
            } => {
                left.validate_with(inputs)?;
                right.validate_with(inputs)?;
                let result = binary_result_type(*operator, left.value_type, right.value_type)?;
                (result == self.value_type)
                    .then_some(())
                    .ok_or(AuthoredExpressionError::ResultTypeMismatch)
            }
        }
    }
}

fn numeric_type(value_type: AuthoredInputType) -> bool {
    matches!(
        value_type,
        AuthoredInputType::Integer | AuthoredInputType::Real | AuthoredInputType::Quantity(_)
    )
}

fn binary_result_type(
    operator: AuthoredNumericOperator,
    left: AuthoredInputType,
    right: AuthoredInputType,
) -> Result<AuthoredInputType, AuthoredExpressionError> {
    use AuthoredInputType::{Integer, Quantity, Real};
    let scalar = |value| matches!(value, Integer | Real);
    match operator {
        AuthoredNumericOperator::Add | AuthoredNumericOperator::Subtract if left == right => {
            Ok(left)
        }
        AuthoredNumericOperator::Add | AuthoredNumericOperator::Subtract
            if scalar(left) && scalar(right) =>
        {
            Ok(Real)
        }
        AuthoredNumericOperator::Multiply if matches!(left, Quantity(_)) && scalar(right) => {
            Ok(left)
        }
        AuthoredNumericOperator::Multiply if scalar(left) && matches!(right, Quantity(_)) => {
            Ok(right)
        }
        AuthoredNumericOperator::Divide if matches!(left, Quantity(_)) && scalar(right) => Ok(left),
        AuthoredNumericOperator::Multiply | AuthoredNumericOperator::Divide
            if scalar(left) && scalar(right) =>
        {
            Ok(Real)
        }
        AuthoredNumericOperator::Multiply
            if matches!((left, right), (Quantity(_), Quantity(_))) =>
        {
            let (Quantity(left), Quantity(right)) = (left, right) else {
                unreachable!()
            };
            Ok(Quantity(left.multiply(right).ok_or(
                AuthoredExpressionError::UnsupportedTypeCombination,
            )?))
        }
        AuthoredNumericOperator::Divide if matches!((left, right), (Quantity(_), Quantity(_))) => {
            let (Quantity(left), Quantity(right)) = (left, right) else {
                unreachable!()
            };
            Ok(Quantity(left.divide(right).ok_or(
                AuthoredExpressionError::UnsupportedTypeCombination,
            )?))
        }
        AuthoredNumericOperator::Divide if scalar(left) && matches!(right, Quantity(_)) => {
            let Quantity(right) = right else {
                unreachable!()
            };
            Ok(Quantity(
                runmat_types::physical::DynamicDimension::from(
                    runmat_types::PhysicalDimension::DIMENSIONLESS,
                )
                .divide(right)
                .ok_or(AuthoredExpressionError::UnsupportedTypeCombination)?,
            ))
        }
        _ => Err(AuthoredExpressionError::UnsupportedTypeCombination),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredExpressionError {
    #[error("authored numeric expression has a non-numeric type")]
    NonNumericType,
    #[error("authored numeric expression has incompatible semantic quantity kinds")]
    WrongQuantityKind,
    #[error("authored numeric expression has a non-finite literal")]
    NonFiniteLiteral,
    #[error("authored numeric expression has an invalid model-input reference")]
    InvalidInputReference,
    #[error("authored numeric expression references unknown model input `{0}`")]
    UnknownInput(String),
    #[error("authored numeric expression input `{0}` has the wrong type")]
    InputTypeMismatch(String),
    #[error("authored numeric expression operator has an unsupported type combination")]
    UnsupportedTypeCombination,
    #[error("authored numeric expression result type is inconsistent with its operands")]
    ResultTypeMismatch,
    #[error("authored numeric expression references unbound model input `{0}`")]
    UnboundInput(String),
    #[error("authored numeric expression input `{0}` is not numeric")]
    NonNumericInput(String),
    #[error("authored numeric expression divides by zero")]
    DivisionByZero,
    #[error("authored numeric expression produced a non-finite result")]
    NonFiniteResult,
}

/// Source-expression arithmetic retains descriptors until the consuming field supplies a kind.
fn evaluate_binary_quantity(
    operator: AuthoredNumericOperator,
    left: runmat_types::physical::DimensionedScalar,
    right: runmat_types::physical::DimensionedScalar,
) -> Result<runmat_types::physical::DimensionedScalar, AuthoredExpressionError> {
    use runmat_types::physical::{QuantityArithmeticError as E, QuantityBinaryOperator as O};
    let operator = match operator {
        AuthoredNumericOperator::Add => O::Add,
        AuthoredNumericOperator::Subtract => O::Subtract,
        AuthoredNumericOperator::Multiply => O::Multiply,
        AuthoredNumericOperator::Divide => O::Divide,
    };
    runmat_types::physical::evaluate_quantity_binary(operator, left, right).map_err(|error| {
        match error {
            E::ResultTypeMismatch => AuthoredExpressionError::ResultTypeMismatch,
            E::WrongQuantityKind => AuthoredExpressionError::WrongQuantityKind,
            E::UnsupportedTypeCombination => AuthoredExpressionError::UnsupportedTypeCombination,
            E::DivisionByZero => AuthoredExpressionError::DivisionByZero,
            E::NonFiniteResult => AuthoredExpressionError::NonFiniteResult,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::PhysicalDimension;

    #[test]
    fn typed_force_expression_evaluates_from_an_exact_model_input() {
        let force = AuthoredInputType::Quantity((PhysicalDimension::FORCE).into());
        let expression = AuthoredNumericExpression {
            value_type: force,
            kind: AuthoredNumericExpressionKind::Negate {
                value: Box::new(AuthoredNumericExpression {
                    value_type: force,
                    kind: AuthoredNumericExpressionKind::ModelInput {
                        input: AuthoredModelInputRef {
                            input_id: "load".into(),
                            value_type: force,
                        },
                    },
                }),
            },
        };
        let definitions = vec![AuthoredInputDefinition {
            input_id: "load".into(),
            value_type: force,
            default: None,
            minimum: None,
            maximum: None,
        }];
        expression.validate(&definitions).unwrap();
        let inputs = BTreeMap::from([(
            "load".into(),
            AuthoredInputValue::Quantity(
                runmat_types::physical::DimensionedScalar::try_from_dimension(
                    PhysicalDimension::FORCE,
                    None,
                    125.0,
                )
                .unwrap(),
            ),
        )]);
        assert_eq!(expression.evaluate(&inputs).unwrap(), -125.0);
    }
}
