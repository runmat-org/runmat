use super::super::{diagnostic, properties::*};
use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic, FeaType, PhysicalDimension,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn validate_members(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    supported: &[&str],
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    let assignments = statements
        .iter()
        .filter(|statement| {
            !matches!(
                statement.kind,
                CheckedFeaStatementKind::GeneratedByFunction { .. }
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    validate_assignments(declaration, &assignments, supported, diagnostics);
    if statements
        .iter()
        .filter(|statement| {
            matches!(
                statement.kind,
                CheckedFeaStatementKind::GeneratedByFunction { .. }
            )
        })
        .count()
        > 1
    {
        diagnostic(
            declaration,
            declaration.span,
            "a spectral input cannot declare more than one generator",
            diagnostics,
        );
    }
}

pub(super) fn assignment<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a CheckedExpression> {
    statements
        .iter()
        .find_map(|statement| match &statement.kind {
            CheckedFeaStatementKind::Assignment { target, value, .. }
                if target.as_slice() == [name] =>
            {
                Some(value)
            }
            _ => None,
        })
}

pub(super) fn enum_assignment<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a str> {
    let expression = required_assignment(declaration, statements, name, diagnostics)?;
    enum_value(declaration, expression, name, diagnostics)
}

pub(super) fn optional_enum<'a>(
    declaration: &CheckedFeaDeclaration,
    statements: &'a [CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<&'a str>> {
    match assignment(statements, name) {
        None => Some(None),
        Some(value) => Some(Some(enum_value(declaration, value, name, diagnostics)?)),
    }
}

fn enum_value<'a>(
    declaration: &CheckedFeaDeclaration,
    expression: &'a CheckedExpression,
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a str> {
    if let CheckedExpressionKind::Enum(value) = &expression.kind {
        Some(value)
    } else {
        diagnostic(
            declaration,
            expression.span,
            format!("spectral member `{name}` requires a closed enum value"),
            diagnostics,
        );
        None
    }
}

pub(super) fn optional_scalar(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<f64>> {
    match assignment(statements, name) {
        None => Some(None),
        Some(value) => Some(Some(scalar(value).or_else(|| {
            diagnostic(
                declaration,
                value.span,
                format!("spectral member `{name}` requires a constant scalar"),
                diagnostics,
            );
            None
        })?)),
    }
}

pub(super) fn vector(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<[f64; 3]> {
    let CheckedExpressionKind::List(values) = &expression.kind else {
        invalid(
            declaration,
            expression,
            name,
            "three numeric components",
            diagnostics,
        );
        return None;
    };
    if values.len() != 3 {
        invalid(
            declaration,
            expression,
            name,
            "three numeric components",
            diagnostics,
        );
        return None;
    }
    Some([
        scalar(&values[0])?,
        scalar(&values[1])?,
        scalar(&values[2])?,
    ])
}

pub(super) fn quantity_vector(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    name: &str,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Vec<f64>> {
    if leaf_type(&expression.value_type) != Some(FeaType::Quantity((dimension).into())) {
        invalid(
            declaration,
            expression,
            name,
            "regular array with compatible units",
            diagnostics,
        );
        return None;
    }
    let values = numerical_tree(expression)?;
    let NumericalTree::List(values) = values else {
        invalid(
            declaration,
            expression,
            name,
            "one-dimensional array",
            diagnostics,
        );
        return None;
    };
    values
        .into_iter()
        .map(|value| match value {
            NumericalTree::Scalar(value) => Some(value),
            NumericalTree::List(_) => None,
        })
        .collect::<Option<Vec<_>>>()
        .or_else(|| {
            invalid(
                declaration,
                expression,
                name,
                "one-dimensional array",
                diagnostics,
            );
            None
        })
}

pub(super) fn quantity_tensor3(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    name: &str,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Vec<Vec<Vec<f64>>>> {
    if leaf_type(&expression.value_type) != Some(FeaType::Quantity((dimension).into())) {
        invalid(
            declaration,
            expression,
            name,
            "rank-three array with compatible units",
            diagnostics,
        );
        return None;
    }
    let NumericalTree::List(samples) = numerical_tree(expression)? else {
        invalid(
            declaration,
            expression,
            name,
            "rank-three array",
            diagnostics,
        );
        return None;
    };
    samples
        .into_iter()
        .map(|sample| {
            let NumericalTree::List(rows) = sample else {
                return None;
            };
            rows.into_iter()
                .map(|row| {
                    let NumericalTree::List(columns) = row else {
                        return None;
                    };
                    columns
                        .into_iter()
                        .map(|value| match value {
                            NumericalTree::Scalar(value) => Some(value),
                            NumericalTree::List(_) => None,
                        })
                        .collect()
                })
                .collect()
        })
        .collect::<Option<Vec<_>>>()
        .or_else(|| {
            invalid(
                declaration,
                expression,
                name,
                "rank-three array",
                diagnostics,
            );
            None
        })
}

pub(super) fn strict_record<'a>(
    declaration: &CheckedFeaDeclaration,
    expression: &'a CheckedExpression,
    name: &str,
    fields: &[&str],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a BTreeMap<String, CheckedExpression>> {
    let CheckedExpressionKind::Record(record) = &expression.kind else {
        invalid(declaration, expression, name, "record", diagnostics);
        return None;
    };
    let allowed = fields.iter().copied().collect::<BTreeSet<_>>();
    for field in record.keys() {
        if !allowed.contains(field.as_str()) {
            diagnostic(
                declaration,
                expression.span,
                format!("unsupported `{name}` record member `{field}`"),
                diagnostics,
            );
        }
    }
    Some(record)
}

pub(super) fn required_record_member<'a>(
    declaration: &CheckedFeaDeclaration,
    record: &'a BTreeMap<String, CheckedExpression>,
    record_name: &str,
    field: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a CheckedExpression> {
    record.get(field).or_else(|| {
        diagnostic(
            declaration,
            declaration.span,
            format!("`{record_name}` record requires `{field}`"),
            diagnostics,
        );
        None
    })
}

pub(super) fn string(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<String> {
    if let CheckedExpressionKind::String(value) = &expression.kind {
        Some(value.clone())
    } else {
        invalid(declaration, expression, name, "string", diagnostics);
        None
    }
}

pub(super) fn enum_expression<'a>(
    declaration: &CheckedFeaDeclaration,
    expression: &'a CheckedExpression,
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<&'a str> {
    enum_value(declaration, expression, name, diagnostics)
}

fn leaf_type(value: &FeaType) -> Option<FeaType> {
    match value {
        FeaType::Vector { element, .. } => leaf_type(element),
        value if value.is_numeric() => Some(value.clone()),
        _ => None,
    }
}

enum NumericalTree {
    Scalar(f64),
    List(Vec<NumericalTree>),
}

fn numerical_tree(expression: &CheckedExpression) -> Option<NumericalTree> {
    match &expression.kind {
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            scale_tree(numerical_tree(value)?, unit.scale(), unit.offset())
        }
        CheckedExpressionKind::List(values) => Some(NumericalTree::List(
            values.iter().map(numerical_tree).collect::<Option<_>>()?,
        )),
        _ => Some(NumericalTree::Scalar(scalar(expression)?)),
    }
}

fn scale_tree(value: NumericalTree, scale: f64, offset: f64) -> Option<NumericalTree> {
    Some(match value {
        NumericalTree::Scalar(value) => NumericalTree::Scalar(value * scale + offset),
        NumericalTree::List(values) => NumericalTree::List(
            values
                .into_iter()
                .map(|value| scale_tree(value, scale, offset))
                .collect::<Option<_>>()?,
        ),
    })
}

fn invalid(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    name: &str,
    expected: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostic(
        declaration,
        expression.span,
        format!("spectral member `{name}` requires a {expected}"),
        diagnostics,
    );
}
