use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic, FeaDiagnosticLocation, FeaDiagnosticPhase, FeaType,
    PhysicalDimension,
};

pub fn block<'a>(
    statements: &'a [CheckedFeaStatement],
    name: &str,
) -> Option<&'a [CheckedFeaStatement]> {
    statements.iter().find_map(|statement| {
        let CheckedFeaStatementKind::Block {
            header, statements, ..
        } = &statement.kind
        else {
            return None;
        };
        (header.trim() == name).then_some(statements.as_slice())
    })
}

pub fn string_assignment(statements: &[CheckedFeaStatement], name: &str) -> Option<String> {
    let value = assignment(statements, name)?;
    match &value.kind {
        CheckedExpressionKind::String(value) => Some(value.clone()),
        _ => None,
    }
}

pub fn optional_quantity(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    assignment(statements, name)
        .and_then(|value| quantity(declaration, name, value, dimension, diagnostics))
}

pub fn required_quantity(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let Some(value) = assignment(statements, name) else {
        missing(declaration, name, diagnostics);
        return None;
    };
    quantity(declaration, name, value, dimension, diagnostics)
}

pub fn required_ratio(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    name: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    let Some(value) = assignment(statements, name) else {
        missing(declaration, name, diagnostics);
        return None;
    };
    if !matches!(value.value_type, FeaType::Integer | FeaType::Real) {
        invalid(
            declaration,
            name,
            "dimensionless scalar",
            value,
            diagnostics,
        );
        return None;
    }
    canonical_scalar(value)
}

fn quantity(
    declaration: &CheckedFeaDeclaration,
    name: &str,
    value: &CheckedExpression,
    dimension: PhysicalDimension,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<f64> {
    if value.value_type != FeaType::Quantity(dimension) {
        invalid(
            declaration,
            name,
            "quantity with the required physical dimension",
            value,
            diagnostics,
        );
        return None;
    }
    canonical_scalar(value)
}

fn assignment<'a>(
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

pub(super) fn typed_quantity(
    value: &CheckedExpression,
    dimension: PhysicalDimension,
) -> Option<f64> {
    (value.value_type == FeaType::Quantity(dimension)).then(|| canonical_scalar(value))?
}

fn canonical_scalar(value: &CheckedExpression) -> Option<f64> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse().ok(),
        CheckedExpressionKind::CadParameterValue(source) => Some(source.value_si()),
        CheckedExpressionKind::Quantity {
            value,
            scale_bits,
            offset_bits,
        } => Some(
            canonical_scalar(value)? * f64::from_bits(*scale_bits) + f64::from_bits(*offset_bits),
        ),
        CheckedExpressionKind::Unary { operator, operand } => {
            let value = canonical_scalar(operand)?;
            Some(match operator {
                runmat_parser::fea::FeaUnaryOperator::Positive => value,
                runmat_parser::fea::FeaUnaryOperator::Negative => -value,
            })
        }
        _ => None,
    }
}

fn missing(declaration: &CheckedFeaDeclaration, name: &str, diagnostics: &mut Vec<FeaDiagnostic>) {
    diagnostics.push(diagnostic(
        declaration,
        declaration.span,
        format!("material capability requires `{name}`"),
    ));
}

fn invalid(
    declaration: &CheckedFeaDeclaration,
    name: &str,
    expected: &str,
    value: &CheckedExpression,
    diagnostics: &mut Vec<FeaDiagnostic>,
) {
    diagnostics.push(diagnostic(
        declaration,
        value.span,
        format!("material property `{name}` requires a {expected}"),
    ));
}

fn diagnostic(
    declaration: &CheckedFeaDeclaration,
    span: runmat_types::Span,
    message: impl Into<String>,
) -> FeaDiagnostic {
    FeaDiagnostic::error(
        "fea.semantic.material",
        message,
        FeaDiagnosticPhase::Lowering,
        Some(FeaDiagnosticLocation {
            path: declaration.module.clone(),
            span,
        }),
    )
}
