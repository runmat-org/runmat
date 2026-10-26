use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryParameterDefinition, GeometryParameterType,
    GeometryScalarExpression, GeometryScalarOperator,
};

use super::GeometryFeatureError;
use runmat_types::physical::{
    evaluate_quantity_binary, DimensionedScalar, DynamicDimension, QuantityBinaryOperator,
    QuantityKind,
};
use runmat_types::PhysicalDimension;

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedGeometryParameters(
    BTreeMap<String, GeometryAuthoringValue>,
    BTreeMap<String, GeometryParameterType>,
);

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
                GeometryAuthoringValue::Scalar(GeometryScalarExpression::Canonical {
                    value: self.quantity(value)?,
                })
            }
            GeometryAuthoringValue::Vector(values) => GeometryAuthoringValue::Vector(Box::new([
                GeometryScalarExpression::Canonical {
                    value: self.quantity(&values[0])?,
                },
                GeometryScalarExpression::Canonical {
                    value: self.quantity(&values[1])?,
                },
                GeometryScalarExpression::Canonical {
                    value: self.quantity(&values[2])?,
                },
            ])),
            GeometryAuthoringValue::Matrix(values) => GeometryAuthoringValue::Matrix(
                values
                    .iter()
                    .map(|value| {
                        Ok(GeometryScalarExpression::Canonical {
                            value: self.quantity(value)?,
                        })
                    })
                    .collect::<Result<_, GeometryFeatureError>>()?,
            ),
            GeometryAuthoringValue::Parameter(name) => {
                let bound = self
                    .get(name)
                    .ok_or_else(|| GeometryFeatureError::MissingParameter(name.clone()))?;
                if matches!(bound, GeometryAuthoringValue::Scalar(_)) {
                    GeometryAuthoringValue::Scalar(GeometryScalarExpression::Canonical {
                        value: self.quantity(&GeometryScalarExpression::Parameter {
                            name: name.clone(),
                        })?,
                    })
                } else if matches!(bound, GeometryAuthoringValue::Vector(_)) {
                    GeometryAuthoringValue::Vector(Box::new([
                        GeometryScalarExpression::Canonical {
                            value: self.quantity(
                                &GeometryScalarExpression::ParameterComponent {
                                    name: name.clone(),
                                    component: 0,
                                },
                            )?,
                        },
                        GeometryScalarExpression::Canonical {
                            value: self.quantity(
                                &GeometryScalarExpression::ParameterComponent {
                                    name: name.clone(),
                                    component: 1,
                                },
                            )?,
                        },
                        GeometryScalarExpression::Canonical {
                            value: self.quantity(
                                &GeometryScalarExpression::ParameterComponent {
                                    name: name.clone(),
                                    component: 2,
                                },
                            )?,
                        },
                    ]))
                } else {
                    self.resolve_value(bound)?
                }
            }
            value => value.clone(),
        })
    }

    fn quantity(
        &self,
        expression: &GeometryScalarExpression,
    ) -> Result<DimensionedScalar, GeometryFeatureError> {
        evaluate_scalar(expression, &self.0, &self.1, &mut BTreeSet::new())
    }

    /// Canonical graph Literal values historically use the owning feature's SI
    /// role. Only that explicit legacy node receives this migration; authored
    /// quantity/parameter expressions must carry a compatible descriptor.
    fn expected(
        &self,
        expression: &GeometryScalarExpression,
        descriptor: DynamicDimension,
    ) -> Result<DimensionedScalar, GeometryFeatureError> {
        if legacy_canonical_numeric(expression) {
            return DimensionedScalar::try_from_descriptor(
                descriptor,
                self.quantity(expression)?.canonical_value(),
            )
            .map_err(unit_error);
        }
        let value = self.quantity(expression)?;
        if !value.descriptor().is_compatible_with(descriptor) {
            return Err(GeometryFeatureError::InvalidPlan(
                "geometry expression has incompatible physical units".into(),
            ));
        }
        Ok(value)
    }
    pub(crate) fn length(
        &self,
        expression: &GeometryScalarExpression,
    ) -> Result<f64, GeometryFeatureError> {
        Ok(self
            .expected(
                expression,
                DynamicDimension::quantity_type(QuantityKind::Length),
            )?
            .canonical_value())
    }
    pub(crate) fn angle(
        &self,
        expression: &GeometryScalarExpression,
    ) -> Result<f64, GeometryFeatureError> {
        Ok(self
            .expected(
                expression,
                DynamicDimension::quantity_type(QuantityKind::Angle),
            )?
            .canonical_value())
    }
    pub(crate) fn scalar(
        &self,
        expression: &GeometryScalarExpression,
    ) -> Result<f64, GeometryFeatureError> {
        Ok(self
            .expected(expression, PhysicalDimension::DIMENSIONLESS.into())?
            .canonical_value())
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
    pub(crate) fn position(
        &self,
        values: &[GeometryScalarExpression; 3],
    ) -> Result<runmat_types::physical::Position, GeometryFeatureError> {
        Ok(runmat_types::physical::Position::new([
            self.length(&values[0])?,
            self.length(&values[1])?,
            self.length(&values[2])?,
        ]))
    }
    pub(crate) fn matrix(
        &self,
        values: &[GeometryScalarExpression],
    ) -> Result<[f64; 16], GeometryFeatureError> {
        // OCCT feature transform is row-major affine: translation slots 3/7/11
        // are lengths, the 3x3 linear part and homogeneous last row are ratios.
        let resolved = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if matches!(index, 3 | 7 | 11) {
                    self.length(value)
                } else {
                    self.scalar(value)
                }
            })
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

    let types = definitions
        .iter()
        .map(|d| (d.parameter_id.clone(), d.parameter_type))
        .collect();
    let resolved = ResolvedGeometryParameters(values, types);
    for definition in definitions {
        match resolved.get(&definition.parameter_id) {
            Some(GeometryAuthoringValue::Scalar(expression)) => {
                let value = resolved.quantity(&GeometryScalarExpression::Parameter {
                    name: definition.parameter_id.clone(),
                })?;
                let expected = match definition.parameter_type {
                    GeometryParameterType::Length => {
                        DynamicDimension::quantity_type(QuantityKind::Length)
                    }
                    GeometryParameterType::Angle => {
                        DynamicDimension::quantity_type(QuantityKind::Angle)
                    }
                    _ => PhysicalDimension::DIMENSIONLESS.into(),
                };
                if !expected.is_compatible_with(value.descriptor()) {
                    return Err(GeometryFeatureError::ParameterType(
                        definition.parameter_id.clone(),
                    ));
                }
                let _ = expression;
            }
            Some(GeometryAuthoringValue::Vector(components)) => {
                let descriptor = match definition.parameter_type {
                    GeometryParameterType::QuantityVector3(d) => d,
                    _ => PhysicalDimension::DIMENSIONLESS.into(),
                };
                for value in components.iter() {
                    resolved.expected(value, descriptor)?;
                }
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
        GeometryParameterType::Vector3 | GeometryParameterType::QuantityVector3(_) => {
            matches!(value, GeometryAuthoringValue::Vector(_))
        }
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

fn unit_error(error: impl std::fmt::Display) -> GeometryFeatureError {
    GeometryFeatureError::InvalidPlan(format!("geometry quantity: {error}"))
}

fn evaluate_scalar(
    expression: &GeometryScalarExpression,
    values: &BTreeMap<String, GeometryAuthoringValue>,
    types: &BTreeMap<String, GeometryParameterType>,
    visiting: &mut BTreeSet<String>,
) -> Result<DimensionedScalar, GeometryFeatureError> {
    match expression {
        GeometryScalarExpression::Literal { value } => {
            DimensionedScalar::try_from_dimension(PhysicalDimension::DIMENSIONLESS, None, *value)
                .map_err(unit_error)
        }
        GeometryScalarExpression::Canonical { value } => Ok(*value),
        GeometryScalarExpression::SourceMeasure { value } => {
            value.validate().map_err(unit_error)?;
            let descriptor = DynamicDimension::try_with_angle(
                value.dimension(),
                value.kind().angle_power(),
                Some(value.kind()),
            )
            .map_err(unit_error)?;
            DimensionedScalar::try_from_descriptor(descriptor, value.canonical_value())
                .map_err(unit_error)
        }
        GeometryScalarExpression::ParameterComponent { name, component } => {
            if !visiting.insert(name.clone()) {
                return Err(GeometryFeatureError::ParameterCycle(name.clone()));
            }
            let expressions = match values.get(name) {
                Some(GeometryAuthoringValue::Vector(expressions)) => expressions,
                Some(_) => return Err(GeometryFeatureError::ParameterType(name.clone())),
                None => return Err(GeometryFeatureError::MissingParameter(name.clone())),
            };
            let expression = expressions
                .get(*component)
                .ok_or_else(|| GeometryFeatureError::ParameterType(name.clone()))?;
            let mut value = evaluate_scalar(expression, values, types, visiting)?;
            let expected = match types.get(name) {
                Some(GeometryParameterType::QuantityVector3(descriptor)) => *descriptor,
                Some(GeometryParameterType::Vector3) => PhysicalDimension::DIMENSIONLESS.into(),
                _ => return Err(GeometryFeatureError::ParameterType(name.clone())),
            };
            if legacy_canonical_numeric(expression) {
                value = DimensionedScalar::try_from_descriptor(expected, value.canonical_value())
                    .map_err(unit_error)?;
            }
            if !expected.is_compatible_with(value.descriptor()) {
                return Err(GeometryFeatureError::ParameterType(name.clone()));
            }
            visiting.remove(name);
            Ok(value)
        }

        GeometryScalarExpression::Quantity {
            value,
            unit,
            semantic_kind,
            ..
        } => {
            let value = evaluate_scalar(value, values, types, visiting)?;
            if value.dimension() != PhysicalDimension::DIMENSIONLESS || value.angle_power() != 0 {
                return Err(GeometryFeatureError::InvalidPlan(
                    "source-unit operand must be dimensionless".into(),
                ));
            }
            let descriptor = DynamicDimension::from_unit(*unit)
                .with_semantic_kind(*semantic_kind)
                .map_err(unit_error)?;
            DimensionedScalar::try_from_descriptor(
                descriptor,
                unit.to_canonical(value.canonical_value())
                    .map_err(unit_error)?,
            )
            .map_err(unit_error)
        }
        GeometryScalarExpression::Parameter { name } => {
            if !visiting.insert(name.clone()) {
                return Err(GeometryFeatureError::ParameterCycle(name.clone()));
            }
            let expression = match values.get(name) {
                Some(GeometryAuthoringValue::Scalar(expression)) => expression,
                Some(_) => return Err(GeometryFeatureError::ParameterType(name.clone())),
                None => return Err(GeometryFeatureError::MissingParameter(name.clone())),
            };
            let mut value = evaluate_scalar(expression, values, types, visiting)?;
            // Legacy canonical literal bindings are already associated with an
            // explicit parameter type; never infer from a field suffix.
            if legacy_canonical_numeric(expression) {
                let kind = match types.get(name) {
                    Some(GeometryParameterType::Length) => Some(QuantityKind::Length),
                    Some(GeometryParameterType::Angle) => Some(QuantityKind::Angle),
                    _ => None,
                };
                if let Some(kind) = kind {
                    value = DimensionedScalar::try_from_descriptor(
                        DynamicDimension::quantity_type(kind),
                        value.canonical_value(),
                    )
                    .map_err(unit_error)?;
                }
            }
            visiting.remove(name);
            Ok(value)
        }
        GeometryScalarExpression::Negate { value } => {
            let value = evaluate_scalar(value, values, types, visiting)?;
            if value.semantic_kind() == Some(QuantityKind::Temperature) {
                return Err(unit_error("cannot negate absolute temperature"));
            }
            DimensionedScalar::try_from_descriptor(value.descriptor(), -value.canonical_value())
                .map_err(unit_error)
        }
        GeometryScalarExpression::Binary {
            operator,
            left,
            right,
        } => {
            let operator = match operator {
                GeometryScalarOperator::Add => QuantityBinaryOperator::Add,
                GeometryScalarOperator::Subtract => QuantityBinaryOperator::Subtract,
                GeometryScalarOperator::Multiply => QuantityBinaryOperator::Multiply,
                GeometryScalarOperator::Divide => QuantityBinaryOperator::Divide,
            };
            evaluate_quantity_binary(
                operator,
                evaluate_scalar(left, values, types, visiting)?,
                evaluate_scalar(right, values, types, visiting)?,
            )
            .map_err(|error| match error {
                runmat_types::physical::QuantityArithmeticError::DivisionByZero => {
                    GeometryFeatureError::DivisionByZero
                }
                _ => unit_error(error),
            })
        }
    }
}

/// Old graph serialization stored canonical numeric expressions without unit nodes.
/// This compatibility classification excludes every explicit quantity and reference.
fn legacy_canonical_numeric(expression: &GeometryScalarExpression) -> bool {
    match expression {
        GeometryScalarExpression::Literal { .. } => true,
        GeometryScalarExpression::Negate { value } => legacy_canonical_numeric(value),
        GeometryScalarExpression::Binary { left, right, .. } => {
            legacy_canonical_numeric(left) && legacy_canonical_numeric(right)
        }
        _ => false,
    }
}
