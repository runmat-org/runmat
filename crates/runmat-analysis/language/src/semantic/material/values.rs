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

pub fn require_closed_law(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    supported: &str,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<()> {
    let Some(value) = assignment(statements, "law") else {
        missing(declaration, "law", diagnostics);
        return None;
    };
    if matches!(&value.kind, CheckedExpressionKind::String(law) if law == supported) {
        return Some(());
    }
    diagnostics.push(diagnostic(
        declaration,
        value.span,
        crate::semantic::pending::with_pending(
            format!("material plastic law must be the explicit string `\"{supported}\"`"),
            // A bare identifier, as fluid and porous laws are written, is planned.
            matches!(&value.kind, CheckedExpressionKind::Enum(_))
                .then_some(crate::semantic::pending::PLASTIC_LAW_IDENTIFIER),
        ),
    ));
    None
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
    if !(value.value_type).is_quantity((dimension).into()) {
        if let Some((message, label)) =
            crate::semantic::units::unit_mismatch(name, value, dimension)
        {
            diagnostics.push(diagnostic(declaration, value.span, message).with_label(label));
            return None;
        }
        invalid(
            declaration,
            name,
            "quantity with the required physical dimension",
            value,
            diagnostics,
        );
        return None;
    }
    if dimension == PhysicalDimension::TEMPERATURE
        && value
            .validate_quantity_kind(runmat_types::physical::QuantityKind::Temperature)
            .is_err()
    {
        invalid(
            declaration,
            name,
            "absolute reference temperature",
            value,
            diagnostics,
        );
        return None;
    }

    canonical_scalar(value)
}

pub(super) fn assignment<'a>(
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
    ((value.value_type).is_quantity((dimension).into())).then(|| canonical_scalar(value))?
}

fn canonical_scalar(value: &CheckedExpression) -> Option<f64> {
    match &value.kind {
        CheckedExpressionKind::Number(source) => source.parse().ok(),
        CheckedExpressionKind::CadParameterValue(source) => Some(source.value_si()),
        CheckedExpressionKind::Quantity { value, unit, .. } => {
            unit.to_canonical(canonical_scalar(value)?).ok()
        }
        CheckedExpressionKind::Unary { operator, operand } => {
            let value = canonical_scalar(operand)?;
            Some(match operator {
                runmat_fea_syntax::FeaUnaryOperator::Positive => value,
                runmat_fea_syntax::FeaUnaryOperator::Negative => -value,
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
