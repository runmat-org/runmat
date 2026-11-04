use super::super::diagnostic;
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic, FeaType, PhysicalDimension,
};
use runmat_fea_syntax::FeaAssignmentRelation;
use std::collections::BTreeSet;

pub(in crate::semantic::model) fn sole_kind_block<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    label: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<(&'a str, &'a [CheckedFeaStatement])> {
    let blocks = statements
        .iter()
        .filter_map(|statement| {
            let CheckedFeaStatementKind::Block {
                header, statements, ..
            } = &statement.kind
            else {
                return None;
            };
            Some((header.trim(), statements.as_slice()))
        })
        .collect::<Vec<_>>();
    if blocks.len() != 1 || statements.len() != 1 {
        diagnostic(
            declaration,
            declaration.span,
            format!("{label} requires exactly one formulation block"),
            diagnostics,
        );
        None
    } else {
        blocks.into_iter().next()
    }
}

pub(in crate::semantic::model) fn validate_assignments(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    supported: &[&str],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let mut seen = BTreeSet::new();
    for statement in statements {
        let CheckedFeaStatementKind::Assignment {
            target, relation, ..
        } = &statement.kind
        else {
            diagnostic(
                declaration,
                statement.span,
                "model definitions accept only declared assignments",
                diagnostics,
            );
            continue;
        };
        let Some(name) = target.first().filter(|_| target.len() == 1) else {
            diagnostic(
                declaration,
                statement.span,
                "definition member names cannot be qualified",
                diagnostics,
            );
            continue;
        };
        if *relation != FeaAssignmentRelation::Equal {
            diagnostic(
                declaration,
                statement.span,
                "definition members require `=`",
                diagnostics,
            );
        } else if !supported.contains(&name.as_str()) {
            diagnostic(
                declaration,
                statement.span,
                crate::semantic::pending::with_pending(
                    format!("unsupported definition member `{name}`"),
                    crate::semantic::pending::definition_member(name.as_str()),
                ),
                diagnostics,
            );
        } else if !seen.insert(name) {
            diagnostic(
                declaration,
                statement.span,
                format!("duplicate definition member `{name}`"),
                diagnostics,
            );
        }
    }
}

pub(in crate::semantic::model) fn reference_name(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    namespace: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<String> {
    let CheckedExpressionKind::Reference(path) = &value.kind else {
        invalid(declaration, value, namespace, "reference", diagnostics);
        return None;
    };
    match path.as_slice() {
        [actual, name] if actual == namespace => Some(name.clone()),
        _ => {
            invalid(
                declaration,
                value,
                namespace,
                "qualified reference",
                diagnostics,
            );
            None
        }
    }
}

pub(in crate::semantic::model) fn required_assignment<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a CheckedExpression> {
    let value = statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        (target.as_slice() == [name]).then_some(value)
    });
    if value.is_none() {
        diagnostic(
            declaration,
            declaration.span,
            format!("model definition requires `{name}`"),
            diagnostics,
        );
    }
    value
}

pub(in crate::semantic::model) fn optional_property_assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
            return None;
        };
        (target.as_slice() == [name]).then_some(value)
    })
}

pub(in crate::semantic::model) fn required_enum<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a str> {
    let value = required_assignment(declaration, statements, name, diagnostics)?;
    let CheckedExpressionKind::Enum(value) = &value.kind else {
        invalid(declaration, value, name, "enum value", diagnostics);
        return None;
    };
    Some(value)
}

pub(in crate::semantic::model) fn required_quantity(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    dimension: impl Into<runmat_types::physical::DynamicDimension>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let value = required_assignment(declaration, statements, name, diagnostics)?;
    let expected = dimension.into();
    if !(value.value_type).is_quantity(expected) {
        let angle = runmat_types::physical::DynamicDimension::quantity_type(
            runmat_types::physical::QuantityKind::Angle,
        );
        let requirement =
            if expected == angle {
                "quantity with an angle unit (deg or rad)"
            } else {
                "quantity of the required dimension"
            };
        invalid(declaration, value, name, requirement, diagnostics);
        return None;
    }
    scalar(value).or_else(|| {
        invalid(declaration, value, name, "constant quantity", diagnostics);
        None
    })
}

pub(in crate::semantic::model) fn optional_property_quantity(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<f64>> {
    let Some(value) = optional_property_assignment(statements, name) else {
        return Some(None);
    };
    if !(value.value_type).is_quantity((dimension).into()) {
        invalid(
            declaration,
            value,
            name,
            "quantity of the required dimension",
            diagnostics,
        );
        return None;
    }
    scalar(value).map(Some).or_else(|| {
        invalid(declaration, value, name, "constant quantity", diagnostics);
        None
    })
}

pub(in crate::semantic::model) fn optional_property_scalar(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<f64>> {
    optional_property_quantity(
        declaration,
        statements,
        name,
        PhysicalDimension::DIMENSIONLESS,
        diagnostics,
    )
}

pub(in crate::semantic::model) fn required_length(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    required_quantity(
        declaration,
        statements,
        name,
        PhysicalDimension::LENGTH,
        diagnostics,
    )
}

pub(in crate::semantic::model) fn required_fourth_length(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    required_quantity(
        declaration,
        statements,
        name,
        PhysicalDimension::LENGTH.power(4)?,
        diagnostics,
    )
}

pub(in crate::semantic::model) fn required_scalar(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let value = required_assignment(declaration, statements, name, diagnostics)?;
    if !matches!(value.value_type, FeaType::Integer | FeaType::Real) {
        invalid(
            declaration,
            value,
            name,
            "dimensionless scalar",
            diagnostics,
        );
        return None;
    }
    scalar(value)
}

pub(in crate::semantic::model) fn required_vector(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    dimension: impl Into<runmat_types::physical::DynamicDimension>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<[f64; 3]> {
    let dimension = dimension.into();
    let value = required_assignment(declaration, statements, name, diagnostics)?;
    if let CheckedExpressionKind::Quantity { value: inner, .. } = &value.kind {
        if matches!(inner.kind, CheckedExpressionKind::List(_)) {
            diagnostic(
                declaration,
                value.span,
                format!(
                    "definition member `{name}` takes a unit on each component (`[0 m, 0 m, 0 m]`); \
                     one trailing unit on the vector is not accepted yet{}",
                    crate::semantic::pending::marker(crate::semantic::pending::VECTOR_TRAILING_UNIT)
                ),
                diagnostics,
            );
            return None;
        }
    }
    let CheckedExpressionKind::List(values) = &value.kind else {
        invalid(
            declaration,
            value,
            name,
            "three-component vector",
            diagnostics,
        );
        return None;
    };
    if values.len() != 3 {
        invalid(
            declaration,
            value,
            name,
            "three-component vector",
            diagnostics,
        );
        return None;
    }
    let mut result = [0.0; 3];
    for (index, component) in values.iter().enumerate() {
        let expected = if dimension == PhysicalDimension::DIMENSIONLESS.into() {
            matches!(component.value_type, FeaType::Integer | FeaType::Real)
        } else {
            (component.value_type).is_quantity(dimension)
        };
        if !expected {
            invalid(
                declaration,
                component,
                name,
                "vector with compatible units",
                diagnostics,
            );
            return None;
        }
        result[index] = scalar(component)?;
    }
    Some(result)
}

pub(in crate::semantic::model) fn scalar(value: &CheckedExpression) -> Option<f64> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse().ok(),
        CheckedExpressionKind::CadParameterValue(source) => Some(source.value_si()),
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            unit.to_canonical(scalar(value)?).ok()
        }
        CheckedExpressionKind::Unary { operator, operand } => {
            let value = scalar(operand)?;
            Some(match operator {
                runmat_fea_syntax::FeaUnaryOperator::Positive => value,
                runmat_fea_syntax::FeaUnaryOperator::Negative => -value,
            })
        }
        _ => None,
    }
}

fn invalid(
    declaration: &CheckedFeaDeclaration,
    value: &CheckedExpression,
    name: &str,
    expected: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostic(
        declaration,
        value.span,
        format!("definition member `{name}` requires a {expected}"),
        diagnostics,
    );
}
