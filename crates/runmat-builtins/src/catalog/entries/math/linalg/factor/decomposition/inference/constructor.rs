use super::super::{DecompositionKind, DecompositionOption, DecompositionTriangle};
use super::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(matrix) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-DECOMPOSITION-ARITY",
            "expected a coefficient matrix",
            0,
        ));
        return finish_fixed(entry, request, object(), diagnostics);
    };
    if !matches!(
        matrix.kind,
        ValueKindFact::Numeric(_)
            | ValueKindFact::Logical
            | ValueKindFact::Unknown
            | ValueKindFact::Composite(_)
    ) || matrix.shape.known_dims().is_some_and(|dims| {
        dims.iter()
            .skip(2)
            .any(|dimension| dimension.is_some_and(|n| n != 1))
    }) {
        diagnostics.push(argument_error(
            "RM-DECOMPOSITION-MATRIX",
            "expected a full or sparse two-dimensional numeric matrix",
            0,
        ));
    }
    let (kind, options_start) = controls(request, &mut diagnostics);
    if kind.is_some_and(DecompositionKind::requires_square) {
        if let Some(dims) = matrix.shape.known_dims() {
            let rows = dims.first().copied().flatten();
            let columns = dims.get(1).copied().unwrap_or(Some(1));
            if rows.zip(columns).is_some_and(|(m, n)| m != n) {
                diagnostics.push(argument_error(
                    "RM-DECOMPOSITION-SQUARE",
                    "selected decomposition type requires a square matrix",
                    0,
                ));
            }
        }
    }
    validate_options(request, options_start, &mut diagnostics);
    finish_fixed(entry, request, object(), diagnostics)
}

/// Only proven source controls select a branch. Unknown text may name either
/// a type or a name-value option, so its layout remains a Runtime decision.
fn controls(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (Option<DecompositionKind>, usize) {
    if request.arguments.len() == 1 {
        return (Some(DecompositionKind::Auto), 1);
    }
    let Some(text) = request.literals.literal_string_at(1) else {
        if invalid_text(&request.arguments[1]) {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-TYPE",
                "decomposition type or option name must be scalar text",
                1,
            ));
        }
        return (None, request.arguments.len());
    };
    let Some(kind) = DecompositionKind::parse(&text) else {
        if DecompositionOption::parse(&text).is_some() {
            return (Some(DecompositionKind::Auto), 1);
        }
        diagnostics.push(argument_error(
            "RM-DECOMPOSITION-TYPE",
            "unsupported decomposition type or option",
            1,
        ));
        return (None, request.arguments.len());
    };
    if let Some(triangle) = request
        .arguments
        .get(2)
        .and_then(|_| request.literals.literal_string_at(2))
    {
        if DecompositionTriangle::parse(&triangle).is_some() {
            if !kind.accepts_triangle() {
                diagnostics.push(argument_error(
                    "RM-DECOMPOSITION-TRIANGLE",
                    "a triangle selector requires chol or triangular type",
                    2,
                ));
            }
            return (Some(kind), 3);
        }
    } else if kind.accepts_triangle()
        && request
            .arguments
            .get(2)
            .is_some_and(|value| !invalid_text(value))
    {
        // Dynamic text may be a triangle selector or a name-value key. The
        // two layouts have different parity; neither is statically proven.
        return (Some(kind), request.arguments.len());
    }
    (Some(kind), 2)
}

fn validate_options(
    request: &CallRequest,
    start: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !(request.arguments.len() - start).is_multiple_of(2) {
        diagnostics.push(argument_error(
            "RM-DECOMPOSITION-OPTIONS",
            "name-value options require complete pairs",
            start,
        ));
        return;
    }
    for index in (start..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_string_at(index) else {
            if invalid_text(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-DECOMPOSITION-OPTION",
                    "decomposition option name must be scalar text",
                    index,
                ));
            }
            continue;
        };
        let Some(option) = DecompositionOption::parse(&name) else {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-OPTION",
                "unsupported decomposition option",
                index,
            ));
            continue;
        };
        let value = &request.arguments[index + 1];
        if value.shape.is_definitely_nonscalar()
            || !matches!(
                value.kind,
                ValueKindFact::Unknown
                    | ValueKindFact::Composite(_)
                    | ValueKindFact::Logical
                    | ValueKindFact::Numeric(_)
            )
        {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-OPTION-VALUE",
                "decomposition option requires a real scalar",
                index + 1,
            ));
        }
        if option == DecompositionOption::RankTolerance
            && request
                .literals
                .numeric_at(index + 1)
                .is_some_and(|tolerance| tolerance < 0.0 || tolerance.is_nan())
        {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-TOLERANCE",
                "RankTolerance must be nonnegative",
                index + 1,
            ));
        }
        if option == DecompositionOption::RankTolerance
            && request
                .literals
                .literal_args
                .get(index + 1)
                .and_then(LiteralValue::integer_is_exact_binary64)
                == Some(false)
        {
            diagnostics.push(argument_error(
                "RM-DECOMPOSITION-TOLERANCE-EXACTNESS",
                "integer RankTolerance must be exactly representable as double",
                index + 1,
            ));
        }
    }
}

fn invalid_text(value: &ValueFact) -> bool {
    crate::catalog::inference::known_invalid_text_scalar(value)
        || !matches!(
            value.kind,
            ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Unknown
                | ValueKindFact::Composite(_)
        )
}
