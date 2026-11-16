//! Line-parameter outputs of an `electromagnetic line_parameters` step.
use runmat_analysis_core::AuthoredOutputExpression;

use crate::{CheckedExpression, CheckedExpressionKind, CheckedFeaDeclaration, FeaDiagnostic};

/// A line parameter of an `electromagnetic line_parameters` step, `analysis.<step>.<name>` or
/// `analysis.<step>.<name>.<trace>`, resolved against the study's cross-section. `None` when
/// `expression` is not a line-parameter reference; `Some(None)` after a located error.
pub(super) fn line_parameter(
    declaration: &CheckedFeaDeclaration,
    expression: &CheckedExpression,
    section: Option<&runmat_geometry_core::CrossSectionDefinition>,
    diagnostics: &mut Vec<FeaDiagnostic>,
) -> Option<Option<AuthoredOutputExpression>> {
    let CheckedExpressionKind::Reference(path) = &expression.kind else {
        return None;
    };
    let [analysis, step_id, name, rest @ ..] = path.as_slice() else {
        return None;
    };
    if analysis != "analysis" || rest.len() > 1 {
        return None;
    }
    let quantity = runmat_analysis_core::ObservationQuantity::from_line_parameter_name(name)?;
    let mut fail = |message: String| {
        super::super::lower::error(declaration, expression.span, message, diagnostics);
        Some(None)
    };
    let Some(section) = section else {
        return fail(format!(
            "`{name}` is a line parameter: it is published by an `electromagnetic line_parameters` analysis on a `cross_section` geometry"
        ));
    };
    let signals = section
        .traces
        .iter()
        .filter(|trace| trace.role == runmat_geometry_core::CrossSectionTraceRole::Signal)
        .map(|trace| trace.name.as_str())
        .collect::<Vec<_>>();
    let conductor = match (quantity.line_parameter_scope()?, rest) {
        (runmat_analysis_core::LineParameterScope::Conductor, [trace]) => {
            if !signals.contains(&trace.as_str()) {
                return fail(format!(
                    "`{trace}` is not a signal trace of the study's cross-section; its signal traces are {}",
                    signals
                        .iter()
                        .map(|name| format!("`{name}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            Some(trace.clone())
        }
        (runmat_analysis_core::LineParameterScope::Conductor, _) => {
            return fail(format!(
                "`{name}` is a per-line value: name the signal trace, `analysis.{step_id}.{name}.<trace>`"
            ))
        }
        (runmat_analysis_core::LineParameterScope::Pair, []) if signals.len() != 2 => {
            return fail(format!(
                "`{name}` is a pair value: the cross-section needs exactly two signal traces, and it has {}",
                signals.len()
            ))
        }
        (_, []) => None,
        (_, _) => {
            return fail(format!(
                "`{name}` is a value of the whole section: write `analysis.{step_id}.{name}` without a trace"
            ))
        }
    };
    Some(Some(AuthoredOutputExpression::LineParameter {
        step_id: step_id.clone(),
        quantity,
        conductor,
        signal_conductors: signals.len(),
    }))
}
