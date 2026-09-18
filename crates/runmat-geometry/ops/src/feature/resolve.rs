use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryParameterDefinition, GeometryParameterType,
    GeometryScalarExpression, GeometryScalarOperator,
};

use super::GeometryFeatureError;

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedGeometryParameters(BTreeMap<String, GeometryAuthoringValue>);

impl ResolvedGeometryParameters {
    pub fn get(&self, name: &str) -> Option<&GeometryAuthoringValue> {
        self.0.get(name)
    }

    pub fn values(&self) -> &BTreeMap<String, GeometryAuthoringValue> {
        &self.0
    }

    pub fn resolve_value(
        &self,
        value: &GeometryAuthoringValue,
    ) -> Result<GeometryAuthoringValue, GeometryFeatureError> {
        Ok(match value {
            GeometryAuthoringValue::Scalar(value) => {
                GeometryAuthoringValue::Scalar(GeometryScalarExpression::Literal {
                    value: self.scalar(value)?,
                })
            }
            GeometryAuthoringValue::Vector(values) => {
                let values = self.vector(values)?;
                GeometryAuthoringValue::Vector(
                    values.map(|value| GeometryScalarExpression::Literal { value }),
                )
            }
            GeometryAuthoringValue::Matrix(values) => GeometryAuthoringValue::Matrix(
                values
                    .iter()
                    .map(|value| {
                        Ok(GeometryScalarExpression::Literal {
                            value: self.scalar(value)?,
                        })
                    })
                    .collect::<Result<_, GeometryFeatureError>>()?,
            ),
            GeometryAuthoringValue::Parameter(name) => self
                .get(name)
                .ok_or_else(|| GeometryFeatureError::MissingParameter(name.clone()))?
                .clone(),
            value => value.clone(),
        })
    }

    pub(crate) fn scalar(
        &self,
        expression: &GeometryScalarExpression,
    ) -> Result<f64, GeometryFeatureError> {
        let mut visiting = BTreeSet::new();
        evaluate_scalar(expression, &self.0, &mut visiting)
    }

    pub(crate) fn vector(
        &self,
        values: &[GeometryScalarExpression; 3],
    ) -> Result<[f64; 3], GeometryFeatureError> {
        Ok([
            self.scalar(&values[0])?,
            self.scalar(&values[1])?,
            self.scalar(&values[2])?,
        ])
    }

    pub(crate) fn matrix(
        &self,
        values: &[GeometryScalarExpression],
    ) -> Result<[f64; 16], GeometryFeatureError> {
        let resolved = values
            .iter()
            .map(|value| self.scalar(value))
            .collect::<Result<Vec<_>, _>>()?;
        resolved.try_into().map_err(|_| {
            GeometryFeatureError::InvalidPlan("transform matrix must contain 16 values".into())
        })
    }
}

pub fn resolve_geometry_parameters(
    definitions: &[GeometryParameterDefinition],
    bindings: &BTreeMap<String, GeometryAuthoringValue>,
) -> Result<ResolvedGeometryParameters, GeometryFeatureError> {
    let declared = definitions
        .iter()
        .map(|definition| definition.parameter_id.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(unknown) = bindings
        .keys()
        .find(|name| !declared.contains(name.as_str()))
    {
        return Err(GeometryFeatureError::InvalidPlan(format!(
            "binding names undeclared geometry parameter `{unknown}`"
        )));
    }

    let mut values = BTreeMap::new();
    for definition in definitions {
        let value = bindings
            .get(&definition.parameter_id)
            .or(definition.default.as_ref())
            .ok_or_else(|| GeometryFeatureError::MissingParameter(definition.parameter_id.clone()))?
            .clone();
        validate_parameter_type(definition, &value)?;
        values.insert(definition.parameter_id.clone(), value);
    }

    let resolved = ResolvedGeometryParameters(values);
    for definition in definitions {
        match resolved.get(&definition.parameter_id) {
            Some(GeometryAuthoringValue::Scalar(expression)) => {
                resolved.scalar(expression)?;
            }
            Some(GeometryAuthoringValue::Vector(components)) => {
                resolved.vector(components)?;
            }
            _ => {}
        }
    }
    Ok(resolved)
}

fn validate_parameter_type(
    definition: &GeometryParameterDefinition,
    value: &GeometryAuthoringValue,
) -> Result<(), GeometryFeatureError> {
    let matches = match definition.parameter_type {
        GeometryParameterType::Boolean => matches!(value, GeometryAuthoringValue::Boolean(_)),
        GeometryParameterType::Scalar
        | GeometryParameterType::Length
        | GeometryParameterType::Angle => {
            matches!(value, GeometryAuthoringValue::Scalar(_))
        }
        GeometryParameterType::Vector3 => matches!(value, GeometryAuthoringValue::Vector(_)),
        GeometryParameterType::Text => matches!(value, GeometryAuthoringValue::Text(_)),
    };
    if matches {
        Ok(())
    } else {
        Err(GeometryFeatureError::ParameterType(
            definition.parameter_id.clone(),
        ))
    }
}

fn evaluate_scalar(
    expression: &GeometryScalarExpression,
    values: &BTreeMap<String, GeometryAuthoringValue>,
    visiting: &mut BTreeSet<String>,
) -> Result<f64, GeometryFeatureError> {
    let value = match expression {
        GeometryScalarExpression::Literal { value } => *value,
        GeometryScalarExpression::Parameter { name } => {
            if !visiting.insert(name.clone()) {
                return Err(GeometryFeatureError::ParameterCycle(name.clone()));
            }
            let expression = match values.get(name) {
                Some(GeometryAuthoringValue::Scalar(expression)) => expression,
                Some(_) => return Err(GeometryFeatureError::ParameterType(name.clone())),
                None => return Err(GeometryFeatureError::MissingParameter(name.clone())),
            };
            let value = evaluate_scalar(expression, values, visiting)?;
            visiting.remove(name);
            value
        }
        GeometryScalarExpression::Negate { value } => -evaluate_scalar(value, values, visiting)?,
        GeometryScalarExpression::Binary {
            operator,
            left,
            right,
        } => {
            let left = evaluate_scalar(left, values, visiting)?;
            let right = evaluate_scalar(right, values, visiting)?;
            match operator {
                GeometryScalarOperator::Add => left + right,
                GeometryScalarOperator::Subtract => left - right,
                GeometryScalarOperator::Multiply => left * right,
                GeometryScalarOperator::Divide if right == 0.0 => {
                    return Err(GeometryFeatureError::DivisionByZero)
                }
                GeometryScalarOperator::Divide => left / right,
            }
        }
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GeometryFeatureError::NonFiniteValue)
    }
}
