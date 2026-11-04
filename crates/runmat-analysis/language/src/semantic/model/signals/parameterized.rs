use crate::{
    CheckedCallArgument, CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration,
    FeaDiagnostic, FeaType,
};
use runmat_analysis_core::{
    AuthoredInputType, AuthoredModelInputRef, AuthoredNumericExpression,
    AuthoredNumericExpressionKind, AuthoredNumericOperator, AuthoredParameterizedSignal,
    AuthoredSignalExpression, CoordinateFrame, PhysicalQuantityKind, SignalId,
};
use runmat_design_syntax::{FeaBinaryOperator, FeaUnaryOperator};

pub(super) fn depends_on_model_input(value: &CheckedExpression) -> bool {
    match &value.kind {
        CheckedExpressionKind::Reference(path) => {
            matches!(path.as_slice(), [namespace, _] if namespace == "input")
        }
        CheckedExpressionKind::List(values) => values.iter().any(depends_on_model_input),
        CheckedExpressionKind::Unary { operand, .. }
        | CheckedExpressionKind::Quantity { value: operand, .. } => depends_on_model_input(operand),
        CheckedExpressionKind::Binary { left, right, .. } => {
            depends_on_model_input(left) || depends_on_model_input(right)
        }
        CheckedExpressionKind::OperationCall { arguments, .. }
        | CheckedExpressionKind::FunctionCall { arguments, .. } => {
            arguments.iter().any(|argument| {
                let value = match argument {
                    CheckedCallArgument::Positional(value)
                    | CheckedCallArgument::Named { value, .. } => value,
                };
                depends_on_model_input(value)
            })
        }
        CheckedExpressionKind::Record(values) => values.values().any(depends_on_model_input),
        _ => false,
    }
}

pub(super) fn lower(
    declaration: &CheckedFeaDeclaration,
    id: &str,
    quantity: PhysicalQuantityKind,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredParameterizedSignal> {
    if value.validate_quantity_kind(quantity).is_err() {
        invalid(declaration, value, diagnostics);
        return None;
    }
    let descriptor = runmat_types::physical::DynamicDimension::quantity_type(quantity);
    let expected = AuthoredInputType::Quantity(descriptor);
    let (value, frame) = match &value.value_type {
        FeaType::Quantity(actual) if actual.is_compatible_with(descriptor) => (
            AuthoredSignalExpression::Scalar(expression(
                declaration,
                value,
                expected,
                diagnostics,
            )?),
            None,
        ),
        FeaType::Vector {
            element,
            length: Some(3),
        } if element.is_quantity(descriptor) => {
            let values = vector_values(value)?;
            let values = values
                .iter()
                .map(|value| expression(declaration, value, expected, diagnostics))
                .collect::<Option<Vec<_>>>()?
                .try_into()
                .ok()?;
            (
                AuthoredSignalExpression::Vector3(values),
                Some(CoordinateFrame::Global),
            )
        }
        _ => {
            invalid(declaration, value, diagnostics);
            return None;
        }
    };
    Some(AuthoredParameterizedSignal {
        signal_id: SignalId(id.into()),
        quantity,
        frame,
        value,
    })
}

fn vector_values(value: &CheckedExpression) -> Option<&[CheckedExpression]> {
    match &value.kind {
        CheckedExpressionKind::List(values) => Some(values),
        CheckedExpressionKind::OperationCall {
            lowering,
            arguments,
            ..
        } if lowering.as_str() == "fea.geometry.global" && arguments.len() == 1 => {
            let value = match &arguments[0] {
                CheckedCallArgument::Positional(value)
                | CheckedCallArgument::Named { value, .. } => value,
            };
            vector_values(value)
        }
        _ => None,
    }
}

pub(super) fn expression(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    expected: AuthoredInputType,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredNumericExpression> {
    let value_type = if is_zero(value) {
        expected
    } else {
        super::super::super::input::authored_type(&value.value_type)?
    };
    let kind = match &value.kind {
        CheckedExpressionKind::Number(source) => {
            let number = source.parse().ok()?;
            if let AuthoredInputType::Quantity(dimension) = value_type {
                AuthoredNumericExpressionKind::Quantity {
                    value: Box::new(AuthoredNumericExpression {
                        value_type: AuthoredInputType::Real,
                        kind: AuthoredNumericExpressionKind::Literal { value: number },
                    }),
                    unit: runmat_types::physical::UnitDefinition::try_with_angle(
                        dimension.dimension(),
                        1.0,
                        0.0,
                        dimension.angle_power(),
                    )
                    .ok()?,
                    semantic_kind: None,
                    source_unit_symbol: "canonical contextual value".into(),
                }
            } else {
                AuthoredNumericExpressionKind::Literal { value: number }
            }
        }
        CheckedExpressionKind::CadParameterValue(source) => {
            AuthoredNumericExpressionKind::Quantity {
                value: Box::new(AuthoredNumericExpression {
                    value_type: AuthoredInputType::Real,
                    kind: AuthoredNumericExpressionKind::Literal {
                        value: source.measure.source_value(),
                    },
                }),
                unit: *source.measure.source_unit(),
                semantic_kind: value.quantity_kind_hint().ok()?,
                source_unit_symbol: source.measure.source_symbol().into(),
            }
        }
        CheckedExpressionKind::Reference(path) if matches!(path.as_slice(), [namespace, _] if namespace == "input") => {
            AuthoredNumericExpressionKind::ModelInput {
                input: AuthoredModelInputRef {
                    input_id: path[1].clone(),
                    value_type,
                },
            }
        }
        CheckedExpressionKind::Unary { operator, operand } => {
            let operand = expression(declaration, operand, expected, diagnostics)?;
            match operator {
                FeaUnaryOperator::Positive => return Some(operand),
                FeaUnaryOperator::Negative => AuthoredNumericExpressionKind::Negate {
                    value: Box::new(operand),
                },
            }
        }
        CheckedExpressionKind::Binary {
            left,
            operator,
            right,
        } => AuthoredNumericExpressionKind::Binary {
            operator: match operator {
                FeaBinaryOperator::Add => AuthoredNumericOperator::Add,
                FeaBinaryOperator::Subtract => AuthoredNumericOperator::Subtract,
                FeaBinaryOperator::Multiply => AuthoredNumericOperator::Multiply,
                FeaBinaryOperator::Divide => AuthoredNumericOperator::Divide,
                _ => {
                    invalid(declaration, value, diagnostics);
                    return None;
                }
            },
            left: Box::new(expression(declaration, left, expected, diagnostics)?),
            right: Box::new(expression(declaration, right, expected, diagnostics)?),
        },
        CheckedExpressionKind::Quantity {
            value: inner,
            unit,
            source_unit_symbol,
            ..
        } => AuthoredNumericExpressionKind::Quantity {
            value: Box::new(expression(
                declaration,
                inner,
                AuthoredInputType::Real,
                diagnostics,
            )?),
            unit: *unit,
            semantic_kind: value.quantity_kind_hint().ok()?,
            source_unit_symbol: source_unit_symbol.clone(),
        },
        _ => {
            invalid(declaration, value, diagnostics);
            return None;
        }
    };
    Some(AuthoredNumericExpression { value_type, kind })
}

fn is_zero(value: &CheckedExpression) -> bool {
    matches!(&value.kind, CheckedExpressionKind::Number(source) if source.parse::<f64>() == Ok(0.0))
}

fn invalid(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    super::super::diagnostic(
        declaration,
        value.span,
        "parameterized signal values must be pure arithmetic over numeric model inputs",
        diagnostics,
    );
}
