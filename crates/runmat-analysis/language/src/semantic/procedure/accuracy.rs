//! Physics-neutral authored algebraic accuracy; no solver policy is silently rewritten.
use super::{lower::error, values::canonical_scalar};
use crate::{
    CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind, FeaDiagnostic, FeaType,
};
use runmat_analysis_core::{AccuracyTolerance, ErrorContributionKind, ProcedureStep};

pub(super) fn apply(
    declaration: &CheckedFeaDeclaration,
    statements: &[CheckedFeaStatement],
    step: &mut ProcedureStep,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> bool {
    let initial_errors = diagnostics.len();
    let mut seen = false;
    for statement in statements {
        let CheckedFeaStatementKind::Block {
            header,
            header_span,
            statements,
        } = &statement.kind
        else {
            continue;
        };
        if header.split_whitespace().next() != Some("accuracy") {
            continue;
        }
        if header.trim() != "accuracy" || seen {
            error(
                declaration,
                *header_span,
                "accuracy requires one unqualified block",
                diagnostics,
            );
            continue;
        }
        seen = true;
        let [setting] = statements.as_slice() else {
            error(
                declaration,
                *header_span,
                "accuracy requires exactly one algebraic_relative_tolerance assignment",
                diagnostics,
            );
            continue;
        };
        let CheckedFeaStatementKind::Assignment { target, value, .. } = &setting.kind else {
            error(
                declaration,
                setting.span,
                "accuracy accepts only algebraic_relative_tolerance",
                diagnostics,
            );
            continue;
        };
        let tolerance = (target.as_slice() == ["algebraic_relative_tolerance"]
            && matches!(value.value_type, FeaType::Integer | FeaType::Real))
        .then(|| canonical_scalar(value))
        .flatten()
        .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0);
        let Some(tolerance) = tolerance else {
            error(
                declaration,
                setting.span,
                "algebraic_relative_tolerance must be a finite dimensionless constant in (0, 1)",
                diagnostics,
            );
            continue;
        };
        let mut criteria = step
            .accuracy
            .criteria
            .iter_mut()
            .filter(|criterion| criterion.contribution == ErrorContributionKind::Algebraic);
        let Some(criterion) = criteria.next() else {
            error(
                declaration,
                setting.span,
                "procedure has no algebraic accuracy criterion",
                diagnostics,
            );
            continue;
        };
        // Preserve IDs, targets, evidence requirements, mandatory status and budgets.
        // Every algebraic criterion must already be relative; never reinterpret units.
        for criterion in std::iter::once(criterion).chain(criteria) {
            if !matches!(criterion.tolerance, AccuracyTolerance::Relative(_)) {
                error(
                    declaration,
                    setting.span,
                    "algebraic accuracy criterion is not relative",
                    diagnostics,
                );
                break;
            }
            criterion.tolerance = AccuracyTolerance::Relative(tolerance);
        }
    }
    diagnostics.len() == initial_errors
}
