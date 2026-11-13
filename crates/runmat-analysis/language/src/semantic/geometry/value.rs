use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, FeaDiagnostic,
    FeaDiagnosticLocation, FeaDiagnosticPhase, FeaType, PhysicalDimension,
};
use runmat_geometry_core::{
    GeometryAuthoringValue, GeometryParameterDefinition, GeometryParameterType,
    GeometryScalarExpression, GeometryScalarOperator,
};

pub(super) fn parameters(
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<GeometryParameterDefinition> {
    declaration
        .inputs
        .values()
        .filter_map(|input| {
            let Some(parameter_type) = parameter_type(&input.value_type) else {
                error(
                    declaration,
                    input.span,
                    format!("geometry input `{}` has an unsupported type", input.name),
                    diagnostics,
                );
                return None;
            };
            let default = input.default.as_ref().and_then(|value| {
                authoring_value(value).or_else(|| {
                    error(
                        declaration,
                        value.span,
                        format!(
                            "geometry input `{}` has a non-canonical default",
                            input.name
                        ),
                        diagnostics,
                    );
                    None
                })
            });
            Some(GeometryParameterDefinition {
                parameter_id: input.name.clone(),
                parameter_type,
                required: input.required,
                default,
            })
        })
        .collect()
}

pub(super) fn scalar(value: &CheckedExpression) -> Option<GeometryScalarExpression> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => Some(source.parse::<f64>().ok()?.into()),
        CheckedExpressionKind::CadParameterValue(source) => {
            Some(GeometryScalarExpression::SourceMeasure {
                value: source.measure.clone(),
            })
        }
        CheckedExpressionKind::Reference(path)
            if path.len() == 2 && path.first().is_some_and(|part| part == "input") =>
        {
            Some(GeometryScalarExpression::Parameter {
                name: path[1].clone(),
            })
        }
        CheckedExpressionKind::Unary { operator, operand } => {
            let operand = scalar(operand)?;
            Some(match operator {
                runmat_design_syntax::FeaUnaryOperator::Positive => operand,
                runmat_design_syntax::FeaUnaryOperator::Negative => {
                    GeometryScalarExpression::Negate {
                        value: Box::new(operand),
                    }
                }
            })
        }
        CheckedExpressionKind::Binary {
            left,
            operator,
            right,
        } => Some(GeometryScalarExpression::Binary {
            operator: match operator {
                runmat_design_syntax::FeaBinaryOperator::Add => GeometryScalarOperator::Add,
                runmat_design_syntax::FeaBinaryOperator::Subtract => {
                    GeometryScalarOperator::Subtract
                }
                runmat_design_syntax::FeaBinaryOperator::Multiply => {
                    GeometryScalarOperator::Multiply
                }
                runmat_design_syntax::FeaBinaryOperator::Divide => GeometryScalarOperator::Divide,
                _ => return None,
            },
            left: Box::new(scalar(left)?),
            right: Box::new(scalar(right)?),
        }),
        CheckedExpressionKind::Quantity {
            value,
            unit,
            source_unit_symbol,
            ..
        } => Some(GeometryScalarExpression::Quantity {
            value: Box::new(scalar(value)?),
            unit: *unit,
            source_unit_symbol: source_unit_symbol.clone(),
            semantic_kind: if unit.offset() != 0.0 {
                Some(runmat_types::physical::QuantityKind::Temperature)
            } else {
                None
            },
        }),
        _ => None,
    }
}

pub(super) fn vector3(value: &CheckedExpression) -> Option<[GeometryScalarExpression; 3]> {
    let value = match &value.kind {
        CheckedExpressionKind::OperationCall {
            lowering,
            arguments,
            ..
        } if lowering.as_str() == "fea.geometry.global" => argument_values(arguments).next()?,
        _ => value,
    };
    if let CheckedExpressionKind::Reference(path) = &value.kind {
        if let [namespace, name] = path.as_slice() {
            if namespace == "input" {
                return Some(std::array::from_fn(|component| {
                    GeometryScalarExpression::ParameterComponent {
                        name: name.clone(),
                        component,
                    }
                }));
            }
        }
    }
    if let CheckedExpressionKind::Quantity {
        value: components,
        unit,
        source_unit_symbol,
        ..
    } = &value.kind
    {
        let CheckedExpressionKind::List(values) = &components.kind else {
            return None;
        };
        let values = values
            .iter()
            .map(|component| {
                Some(GeometryScalarExpression::Quantity {
                    value: Box::new(scalar(component)?),
                    unit: *unit,
                    source_unit_symbol: source_unit_symbol.clone(),
                    semantic_kind: if unit.offset() != 0.0 {
                        Some(runmat_types::physical::QuantityKind::Temperature)
                    } else {
                        None
                    },
                })
            })
            .collect::<Option<Vec<_>>>()?;
        return values.try_into().ok();
    }
    let CheckedExpressionKind::List(values) = &value.kind else {
        return None;
    };
    let values = values.iter().map(scalar).collect::<Option<Vec<_>>>()?;
    values.try_into().ok()
}

pub(super) fn authoring_value(value: &CheckedExpression) -> Option<GeometryAuthoringValue> {
    match &value.kind {
        CheckedExpressionKind::String(value) => Some(GeometryAuthoringValue::Text(value.clone())),
        CheckedExpressionKind::Boolean(value) => Some(GeometryAuthoringValue::Boolean(*value)),
        CheckedExpressionKind::Reference(path)
            if path.len() == 2 && path.first().is_some_and(|part| part == "input") =>
        {
            Some(GeometryAuthoringValue::Parameter(path[1].clone()))
        }
        CheckedExpressionKind::List(values) if values.len() == 3 => {
            Some(GeometryAuthoringValue::Vector(Box::new(vector3(value)?)))
        }
        CheckedExpressionKind::Quantity {
            value: components, ..
        } if matches!(&components.kind, CheckedExpressionKind::List(values) if values.len() == 3) => {
            Some(GeometryAuthoringValue::Vector(Box::new(vector3(value)?)))
        }
        _ => scalar(value).map(GeometryAuthoringValue::Scalar),
    }
}

fn argument_values(
    arguments: &[crate::CheckedCallArgument],
) -> impl Iterator<Item = &CheckedExpression> {
    arguments.iter().map(|argument| match argument {
        crate::CheckedCallArgument::Positional(value)
        | crate::CheckedCallArgument::Named { value, .. } => value,
    })
}

fn parameter_type(value_type: &FeaType) -> Option<GeometryParameterType> {
    match value_type {
        FeaType::Boolean => Some(GeometryParameterType::Boolean),
        FeaType::Integer | FeaType::Real => Some(GeometryParameterType::Scalar),
        FeaType::Quantity(spec) if spec.is_compatible_with(PhysicalDimension::LENGTH.into()) => {
            Some(GeometryParameterType::Length)
        }
        FeaType::Quantity(spec)
            if spec.is_compatible_with(
                runmat_types::physical::DynamicDimension::quantity_type(
                    runmat_types::physical::QuantityKind::Angle,
                ),
            ) =>
        {
            Some(GeometryParameterType::Angle)
        }
        FeaType::String => Some(GeometryParameterType::Text),
        FeaType::Vector {
            length: Some(3),
            element,
        } => match element.as_ref() {
            FeaType::Quantity(descriptor) => {
                Some(GeometryParameterType::QuantityVector3(*descriptor))
            }
            FeaType::Integer | FeaType::Real => Some(GeometryParameterType::Vector3),
            _ => None,
        },
        _ => None,
    }
}

fn error(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.geometry_parameter",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}
