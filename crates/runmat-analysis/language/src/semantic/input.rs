use super::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, FeaTopologyRole, FeaType,
};
use crate::{
    BoundFeaProgram, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaSymbolKey,
    FeaSymbolKind, FeaSymbolOrigin,
};
use runmat_analysis_core::{
    AuthoredInputDefinition, AuthoredInputType, AuthoredInputValue, AuthoredMaterialRef,
};
use runmat_fea_syntax::{FeaBinaryOperator, FeaUnaryOperator};
use runmat_geometry_core::{GeometryRegionRef, GeometryRegionRole};

pub(super) fn lower_definitions(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Vec<AuthoredInputDefinition> {
    declaration
        .inputs
        .values()
        .filter_map(|input| {
            let Some(value_type) = authored_type(&input.value_type) else {
                diagnostic(
                    declaration,
                    input.span,
                    format!("input `{}` has no canonical authored type", input.name),
                    diagnostics,
                );
                return None;
            };
            let initial_errors = diagnostics.len();
            let default = input
                .default
                .as_ref()
                .and_then(|value| lower_value(bound, declaration, value_type, value, diagnostics));
            let minimum = input
                .minimum
                .as_ref()
                .and_then(|value| lower_value(bound, declaration, value_type, value, diagnostics));
            let maximum = input
                .maximum
                .as_ref()
                .and_then(|value| lower_value(bound, declaration, value_type, value, diagnostics));
            if diagnostics.len() != initial_errors {
                return None;
            }
            let definition = AuthoredInputDefinition {
                input_id: input.name.clone(),
                value_type,
                default,
                minimum,
                maximum,
            };
            if let Err(error) = definition.validate() {
                diagnostic(
                    declaration,
                    input.span,
                    format!("input `{}` is invalid: {error}", input.name),
                    diagnostics,
                );
                return None;
            }
            Some(definition)
        })
        .collect()
}

pub(super) fn authored_type(value_type: &FeaType) -> Option<AuthoredInputType> {
    Some(match value_type {
        FeaType::Boolean => AuthoredInputType::Boolean,
        FeaType::Integer => AuthoredInputType::Integer,
        FeaType::Real => AuthoredInputType::Real,
        FeaType::Complex => AuthoredInputType::Complex,
        FeaType::String => AuthoredInputType::String,
        FeaType::Quantity(dimension) => AuthoredInputType::Quantity(*dimension),
        FeaType::Declaration(FeaSymbolKind::Material) => AuthoredInputType::Material,
        FeaType::Region(role) => AuthoredInputType::Region(geometry_role(*role)),
        _ => return None,
    })
}

pub(super) fn lower_value(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expected: AuthoredInputType,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredInputValue> {
    let actual = authored_type(&expression.value_type);
    if !actual.is_some_and(|actual| type_assignable(expected, actual)) {
        return invalid_value(declaration, expression, expected, diagnostics);
    }
    let value = match expected {
        AuthoredInputType::Boolean => match expression.kind {
            CheckedExpressionKind::Boolean(value) => AuthoredInputValue::Boolean(value),
            _ => return invalid_value(declaration, expression, expected, diagnostics),
        },
        AuthoredInputType::Integer => AuthoredInputValue::Integer(
            integer(expression)
                .or_else(|| invalid_value(declaration, expression, expected, diagnostics))?,
        ),
        AuthoredInputType::Real => AuthoredInputValue::Real(
            scalar(expression)
                .or_else(|| invalid_value(declaration, expression, expected, diagnostics))?,
        ),
        AuthoredInputType::Complex => AuthoredInputValue::Complex {
            real: scalar(expression)
                .or_else(|| invalid_value(declaration, expression, expected, diagnostics))?,
            imaginary: 0.0,
        },
        AuthoredInputType::String => match &expression.kind {
            CheckedExpressionKind::String(value) => AuthoredInputValue::String(value.clone()),
            _ => return invalid_value(declaration, expression, expected, diagnostics),
        },
        AuthoredInputType::Quantity(dimension) => {
            let kind = expression
                .quantity_kind_hint()
                .ok()
                .or_else(|| invalid_value(declaration, expression, expected, diagnostics))?;
            let value = scalar(expression)
                .or_else(|| invalid_value(declaration, expression, expected, diagnostics))?;
            AuthoredInputValue::Quantity(
                runmat_types::physical::DimensionedScalar::try_from_descriptor(
                    dimension
                        .with_semantic_kind(kind.or(dimension.kind()))
                        .ok()?,
                    value,
                )
                .ok()
                .or_else(|| invalid_value(declaration, expression, expected, diagnostics))?,
            )
        }
        AuthoredInputType::Material => AuthoredInputValue::Material(material_reference(
            bound,
            declaration,
            expression,
            diagnostics,
        )?),
        AuthoredInputType::Region(role) => AuthoredInputValue::Region(region_reference(
            bound,
            declaration,
            expression,
            role,
            diagnostics,
        )?),
    };
    if let Err(error) = value.validate() {
        diagnostic(declaration, expression.span, error.to_string(), diagnostics);
        return None;
    }
    Some(value)
}

fn type_assignable(expected: AuthoredInputType, actual: AuthoredInputType) -> bool {
    expected.accepts_type(actual)
        || matches!(
            (expected, actual),
            (AuthoredInputType::Real, AuthoredInputType::Integer)
                | (
                    AuthoredInputType::Complex,
                    AuthoredInputType::Integer | AuthoredInputType::Real
                )
        )
        // A bare number is a dimensionless quantity.
        || matches!(
            (expected, actual),
            (
                AuthoredInputType::Quantity(dimension),
                AuthoredInputType::Integer | AuthoredInputType::Real
            ) if dimension.dimension() == runmat_types::PhysicalDimension::DIMENSIONLESS
        )
}

fn integer(expression: &CheckedExpression) -> Option<i64> {
    match &expression.kind {
        CheckedExpressionKind::Number(value) => value.parse().ok(),
        CheckedExpressionKind::Unary { operator, operand } => {
            let value = integer(operand)?;
            match operator {
                FeaUnaryOperator::Positive => Some(value),
                FeaUnaryOperator::Negative => value.checked_neg(),
            }
        }
        _ => None,
    }
}

fn scalar(expression: &CheckedExpression) -> Option<f64> {
    match &expression.kind {
        CheckedExpressionKind::Number(value) => value.parse().ok(),
        CheckedExpressionKind::CadParameterValue(value) => Some(value.value_si()),
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            unit.to_canonical(scalar(value)?).ok()
        }
        CheckedExpressionKind::Unary { operator, operand } => {
            let value = scalar(operand)?;
            Some(match operator {
                FeaUnaryOperator::Positive => value,
                FeaUnaryOperator::Negative => -value,
            })
        }
        CheckedExpressionKind::Binary {
            left,
            operator,
            right,
        } => {
            let left = scalar(left)?;
            let right = scalar(right)?;
            match operator {
                FeaBinaryOperator::Add => Some(left + right),
                FeaBinaryOperator::Subtract => Some(left - right),
                FeaBinaryOperator::Multiply => Some(left * right),
                FeaBinaryOperator::Divide => Some(left / right),
                _ => None,
            }
        }
        _ => None,
    }
}

fn material_reference(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<AuthoredMaterialRef> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        return invalid_value(
            declaration,
            expression,
            AuthoredInputType::Material,
            diagnostics,
        );
    };
    let (module_path, material_id) = declaration_reference(
        bound,
        declaration,
        path,
        FeaSymbolKind::Material,
        expression,
        diagnostics,
    )?;
    Some(AuthoredMaterialRef {
        module_path,
        material_id,
    })
}

fn region_reference(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    expected_role: GeometryRegionRole,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<GeometryRegionRef> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        return invalid_value(
            declaration,
            expression,
            AuthoredInputType::Region(expected_role),
            diagnostics,
        );
    };
    let [namespace, geometry, region_namespace, region] = path.as_slice() else {
        return invalid_value(
            declaration,
            expression,
            AuthoredInputType::Region(expected_role),
            diagnostics,
        );
    };
    if namespace != "geometry" || region_namespace != "region" {
        return invalid_value(
            declaration,
            expression,
            AuthoredInputType::Region(expected_role),
            diagnostics,
        );
    }
    let (module_path, geometry_id) = declaration_reference(
        bound,
        declaration,
        &[namespace.clone(), geometry.clone()],
        FeaSymbolKind::Geometry,
        expression,
        diagnostics,
    )?;
    Some(GeometryRegionRef {
        module_path,
        geometry_id,
        region_id: region.clone(),
        role: expected_role,
    })
}

fn declaration_reference(
    bound: &BoundFeaProgram,
    declaration: &CheckedFeaDeclaration,
    path: &[String],
    kind: FeaSymbolKind,
    expression: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(String, String)> {
    let [_, name] = path else {
        return invalid_value(
            declaration,
            expression,
            AuthoredInputType::String,
            diagnostics,
        );
    };
    let key = FeaSymbolKey {
        kind,
        name: name.clone(),
    };
    let symbol = bound
        .symbols
        .get(&declaration.module)
        .and_then(|symbols| symbols.get(&key));
    match symbol.map(|symbol| &symbol.origin) {
        Some(FeaSymbolOrigin::Local) => {
            Some((declaration.module.as_str().to_string(), name.clone()))
        }
        Some(FeaSymbolOrigin::Import {
            target_module,
            target,
        }) => Some((target_module.as_str().to_string(), target.name.clone())),
        _ => {
            diagnostic(
                declaration,
                expression.span,
                format!("referenced {:?} declaration is unavailable", kind),
                diagnostics,
            );
            None
        }
    }
}

fn geometry_role(role: FeaTopologyRole) -> GeometryRegionRole {
    match role {
        FeaTopologyRole::Vertex => GeometryRegionRole::Vertex,
        FeaTopologyRole::Edge => GeometryRegionRole::Edge,
        FeaTopologyRole::Face => GeometryRegionRole::Face,
        FeaTopologyRole::Solid => GeometryRegionRole::Solid,
        FeaTopologyRole::Cell => GeometryRegionRole::Cell,
    }
}

fn invalid_value<T>(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    expected: AuthoredInputType,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<T> {
    let found = match &expression.kind {
        CheckedExpressionKind::Quantity {
            source_unit_symbol, ..
        } => match expression.value_type.quantity_descriptor() {
            Some(found) => format!(
                "; found {} written in {source_unit_symbol}",
                super::units::describe_quantity_type(found)
            ),
            None => String::new(),
        },
        _ => String::new(),
    };
    diagnostic(
        declaration,
        expression.span,
        format!(
            "input value must be a constant {}{found}",
            describe_input_type(&expected)
        ),
        diagnostics,
    );
    None
}

/// The input type as the `.fea` author writes it.
fn describe_input_type(expected: &AuthoredInputType) -> String {
    match expected {
        AuthoredInputType::Boolean => "boolean".into(),
        AuthoredInputType::Integer => "integer".into(),
        AuthoredInputType::Real => "real number".into(),
        AuthoredInputType::Complex => "complex number".into(),
        AuthoredInputType::String => "string".into(),
        AuthoredInputType::Quantity(quantity) => super::units::describe_quantity_type(*quantity),
        AuthoredInputType::Material => "material reference".into(),
        AuthoredInputType::Region(role) => format!("{} region reference", region_role_name(*role)),
    }
}

fn region_role_name(role: GeometryRegionRole) -> &'static str {
    match role {
        GeometryRegionRole::Vertex => "vertex",
        GeometryRegionRole::Edge => "edge",
        GeometryRegionRole::Face => "face",
        GeometryRegionRole::Solid => "solid",
        GeometryRegionRole::Cell => "cell",
    }
}

fn diagnostic(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(FeaDiagnostic::error(
        "fea.semantic.input",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    ));
}
