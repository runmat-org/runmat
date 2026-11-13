use crate::{
    CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement,
    CheckedFeaStatementKind, FeaDiagnostic,
};
use runmat_analysis_core::{SignalExtrapolation, SignalInterpolation};

use super::super::diagnostic;

pub(super) fn interpolation(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalInterpolation> {
    let Some(value) = assignment(statements, "interpolation") else {
        diagnostic(
            declaration,
            declaration.span,
            "sampled signal requires an interpolation policy",
            diagnostics,
        );
        return None;
    };
    match enum_name(value) {
        Some("linear") => Some(SignalInterpolation::Linear),
        Some("left_endpoint") => Some(SignalInterpolation::LeftEndpoint),
        Some("right_endpoint") => Some(SignalInterpolation::RightEndpoint),
        _ => {
            diagnostic(
                declaration,
                value.span,
                "unsupported signal interpolation policy",
                diagnostics,
            );
            None
        }
    }
}

pub(super) fn extrapolation(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<SignalExtrapolation> {
    let Some(value) = assignment(statements, "extrapolation") else {
        return Some(SignalExtrapolation::Reject);
    };
    match enum_name(value) {
        Some("reject") => Some(SignalExtrapolation::Reject),
        Some("hold_endpoint") => Some(SignalExtrapolation::HoldEndpoint),
        Some("zero") => Some(SignalExtrapolation::Zero),
        _ => {
            diagnostic(
                declaration,
                value.span,
                "unsupported signal extrapolation policy",
                diagnostics,
            );
            None
        }
    }
}

pub(super) fn enum_name(value: &CheckedExpression) -> Option<&str> {
    match &value.kind {
        CheckedExpressionKind::Enum(name) => Some(name),
        _ => None,
    }
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
