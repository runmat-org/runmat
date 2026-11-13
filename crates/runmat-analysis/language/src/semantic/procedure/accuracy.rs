//! Physics-neutral authored algebraic accuracy; no solver policy is silently rewritten.
//!
//! `accuracy { algebraic_measure = backward_error | relative_residual;
//! algebraic_relative_tolerance = <number> }`: the measure moves the step's algebraic criterion
//! to that measure at its default tolerance, then an authored tolerance replaces it.
use super::{lower::error, values::canonical_scalar};
use crate::{
    CheckedExpressionKind, CheckedFeaDeclaration, CheckedFeaStatement, CheckedFeaStatementKind,
    FeaDiagnostic, FeaType,
};
use runmat_analysis_core::{
    AccuracyTolerance, AlgebraicAccuracyMeasure, ErrorContributionKind, ProcedureStep,
};

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
        let mut measure_setting = None;
        let mut tolerance_setting = None;
        let mut valid = !statements.is_empty();
        for setting in statements {
            let slot = match &setting.kind {
                CheckedFeaStatementKind::Assignment { target, value, .. } => {
                    match target.as_slice() {
                        [name] if name == "algebraic_measure" => {
                            Some((&mut measure_setting, value))
                        }
                        [name] if name == "algebraic_relative_tolerance" => {
                            Some((&mut tolerance_setting, value))
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            match slot {
                Some((slot, value)) if slot.is_none() => *slot = Some((setting.span, value)),
                _ => {
                    error(
                        declaration,
                        setting.span,
                        "accuracy accepts one `algebraic_measure` and one `algebraic_relative_tolerance`",
                        diagnostics,
                    );
                    valid = false;
                }
            }
        }
        if statements.is_empty() {
            error(
                declaration,
                *header_span,
                "accuracy requires `algebraic_measure`, `algebraic_relative_tolerance` or both",
                diagnostics,
            );
        }
        if !valid {
            continue;
        }
        if let Some((span, value)) = measure_setting {
            let measure = match &value.kind {
                CheckedExpressionKind::Enum(name) => algebraic_measure(name),
                _ => None,
            };
            let Some(measure) = measure else {
                error(
                    declaration,
                    span,
                    "`algebraic_measure` must be `backward_error` or `relative_residual`",
                    diagnostics,
                );
                continue;
            };
            if measure == AlgebraicAccuracyMeasure::BackwardError
                && !ProcedureStep::qualifies_on_backward_error(step.family, step.procedure)
            {
                error(
                    declaration,
                    span,
                    "`backward_error` is published by structural modal steps; this procedure qualifies on `relative_residual`",
                    diagnostics,
                );
                continue;
            }
            if !step.accuracy.set_algebraic_measure(measure) {
                error(
                    declaration,
                    span,
                    "procedure has no algebraic residual criterion",
                    diagnostics,
                );
                continue;
            }
        }
        let Some((span, value)) = tolerance_setting else {
            continue;
        };
        let tolerance = matches!(value.value_type, FeaType::Integer | FeaType::Real)
            .then(|| canonical_scalar(value))
            .flatten()
            .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0);
        let Some(tolerance) = tolerance else {
            error(
                declaration,
                span,
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
                span,
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
                    span,
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

/// The closed set of algebraic measure words (the generated grammar reads this table).
fn algebraic_measure(name: &str) -> Option<AlgebraicAccuracyMeasure> {
    match name {
        "backward_error" => Some(AlgebraicAccuracyMeasure::BackwardError),
        "relative_residual" => Some(AlgebraicAccuracyMeasure::RelativeResidual),
        _ => None,
    }
}
