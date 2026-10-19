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

    pub fn evaluate(
        &self,
        inputs: &BTreeMap<String, AuthoredInputValue>,
    ) -> Result<f64, AuthoredExpressionError> {
        let value = match &self.kind {
            AuthoredNumericExpressionKind::Literal { value } => *value,
            AuthoredNumericExpressionKind::ModelInput { input } => {
                let value = inputs
                    .get(&input.input_id)
                    .ok_or_else(|| AuthoredExpressionError::UnboundInput(input.input_id.clone()))?;
                if !value.matches(input.value_type) {
                    return Err(AuthoredExpressionError::InputTypeMismatch(
                        input.input_id.clone(),
                    ));
                }
                numeric_value(value).ok_or_else(|| {
                    AuthoredExpressionError::NonNumericInput(input.input_id.clone())
                })?
            }
            AuthoredNumericExpressionKind::Negate { value } => -value.evaluate(inputs)?,
            AuthoredNumericExpressionKind::Binary {
                operator,
                left,
                right,
            } => {
                let left = left.evaluate(inputs)?;
                let right = right.evaluate(inputs)?;
                match operator {
                    AuthoredNumericOperator::Add => left + right,
                    AuthoredNumericOperator::Subtract => left - right,
                    AuthoredNumericOperator::Multiply => left * right,
                    AuthoredNumericOperator::Divide if right != 0.0 => left / right,
                    AuthoredNumericOperator::Divide => {
                        return Err(AuthoredExpressionError::DivisionByZero)
                    }
                }
            }
        };
        value
            .is_finite()
            .then_some(value)
            .ok_or(AuthoredExpressionError::NonFiniteResult)
    }

    fn validate_with(
        &self,
        inputs: &BTreeMap<&str, AuthoredInputType>,
    ) -> Result<(), AuthoredExpressionError> {
        if !numeric_type(self.value_type) {
            return Err(AuthoredExpressionError::NonNumericType);
        }
        match &self.kind {
            AuthoredNumericExpressionKind::Literal { value } if value.is_finite() => Ok(()),
            AuthoredNumericExpressionKind::Literal { .. } => {
                Err(AuthoredExpressionError::NonFiniteLiteral)
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
        _ => Err(AuthoredExpressionError::UnsupportedTypeCombination),
    }
}

fn numeric_value(value: &AuthoredInputValue) -> Option<f64> {
    match value {
        AuthoredInputValue::Integer(value) => Some(*value as f64),
        AuthoredInputValue::Real(value) => Some(*value),
        AuthoredInputValue::Quantity { value_si, .. } => Some(*value_si),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredExpressionError {
    #[error("authored numeric expression has a non-numeric type")]
    NonNumericType,
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

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_types::QuantityDimension;

    #[test]
    fn typed_force_expression_evaluates_from_an_exact_model_input() {
        let force = AuthoredInputType::Quantity(QuantityDimension::Force);
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
            AuthoredInputValue::Quantity {
                dimension: QuantityDimension::Force,
                value_si: 125.0,
            },
        )]);
        assert_eq!(expression.evaluate(&inputs).unwrap(), -125.0);
    }
}
