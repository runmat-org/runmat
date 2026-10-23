use super::{EigsOption, EIGS_ERROR_INVALID_ARGUMENT, EIGS_ERROR_INVALID_INPUT};
use crate::catalog::entries::math::linalg::factor::inference::{
    dimensions, double_scalar, extent, factor_output, matrix_input, matrix_shape,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, EffectKind, LiteralValue, ValueFact, ValueKindFact,
};

fn text(request: &CallRequest, index: usize) -> Option<&str> {
    match request.literals.literal_args.get(index)? {
        LiteralValue::Character(v) | LiteralValue::String(v) | LiteralValue::Keyword(v) => {
            Some(v.trim())
        }
        _ => None,
    }
}

fn structural(request: &CallRequest, index: usize) -> Option<usize> {
    let literal = request.literals.literal_args.get(index)?;
    if let Some(value) = literal.integer_value_i128() {
        return usize::try_from(value).ok().filter(|&n| n > 0);
    }
    request
        .literals
        .numeric_at(index)
        .filter(|n| n.is_finite() && *n > 0.0 && n.fract() == 0.0 && *n < usize::MAX as f64)
        .map(|n| n as usize)
}
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let matrix = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    let callable = matches!(&matrix.kind, ValueKindFact::Callable(_));
    let mut diagnostics = if callable {
        Vec::new()
    } else {
        matrix_input(
            request,
            usize::MAX,
            true,
            EIGS_ERROR_INVALID_ARGUMENT.code,
            EIGS_ERROR_INVALID_INPUT.code,
        )
    };
    let n = if callable {
        structural(request, 1)
    } else {
        extent(&matrix)
    };
    let mut cursor = if callable { 2 } else { 1 };
    if callable && request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            EIGS_ERROR_INVALID_ARGUMENT.code,
            "a callable eigs operator requires its matrix order",
            1,
        ));
    }
    if let Some(other) = request.arguments.get(cursor) {
        if matches!(&other.kind, ValueKindFact::Numeric(_)) && !other.is_scalar() {
            if let Some((Some(rows), Some(columns))) = dimensions(other) {
                if (rows != 0 || columns != 0) && (rows != columns || n.is_some_and(|n| n != rows))
                {
                    diagnostics.push(argument_error(
                        EIGS_ERROR_INVALID_INPUT.code,
                        "generalized eigenproblem matrices must have equal square shape",
                        cursor,
                    ));
                }
            }
            cursor += 1;
        }
    }
    let mut k = n.map(|n| n.min(6));
    if request
        .arguments
        .get(cursor)
        .is_some_and(|fact| matches!(fact.kind, ValueKindFact::Numeric(_)) && fact.is_scalar())
    {
        k = structural(request, cursor).map(|k| n.map_or(k, |n| k.min(n)));
        cursor += 1;
    }
    // A selector may precede options; a typed option key may start directly.
    if request.arguments.get(cursor).is_some_and(|fact| {
        matches!(fact.kind, ValueKindFact::Numeric(_)) || text(request, cursor).is_some()
    }) && text(request, cursor)
        .and_then(EigsOption::from_name)
        .is_none()
    {
        cursor += 1;
    }
    let mut known_options = (request.arguments.len().saturating_sub(cursor)).is_multiple_of(2);
    let mut display = Some(false);
    if known_options {
        for index in (cursor..request.arguments.len()).step_by(2) {
            let Some(key) = text(request, index).and_then(EigsOption::from_name) else {
                known_options = false;
                break;
            };
            let value = index + 1;
            match key {
                EigsOption::Failure => match text(request, value) {
                    Some(v)
                        if v.eq_ignore_ascii_case("keep")
                            || v.eq_ignore_ascii_case("replacenan") => {}
                    Some(v) if v.eq_ignore_ascii_case("drop") => k = None,
                    Some(_) => diagnostics.push(argument_error(
                        EIGS_ERROR_INVALID_ARGUMENT.code,
                        "FailureTreatment must be keep, drop, or replacenan",
                        value,
                    )),
                    None => k = None,
                },
                EigsOption::Display => {
                    display = request
                        .literals
                        .literal_bool_at(value)
                        .or_else(|| request.literals.numeric_at(value).map(|n| n != 0.0))
                }
                EigsOption::Tolerance => {
                    if request
                        .literals
                        .numeric_at(value)
                        .is_some_and(|n| !n.is_finite() || n <= 0.0)
                    {
                        diagnostics.push(argument_error(
                            EIGS_ERROR_INVALID_ARGUMENT.code,
                            "Tolerance must be a positive finite scalar",
                            value,
                        ));
                    }
                }
                EigsOption::Iterations | EigsOption::Subspace => {
                    if request.literals.numeric_at(value).is_some()
                        && structural(request, value).is_none()
                    {
                        diagnostics.push(argument_error(
                            EIGS_ERROR_INVALID_ARGUMENT.code,
                            "iteration and subspace controls must be positive integers",
                            value,
                        ));
                    }
                }
                EigsOption::Start | EigsOption::Permutation => {
                    if let Some((Some(rows), Some(columns))) =
                        request.arguments.get(value).and_then(dimensions)
                    {
                        if (rows != 1 && columns != 1)
                            || n.is_some_and(|n| rows.checked_mul(columns) != Some(n))
                        {
                            diagnostics.push(argument_error(
                                EIGS_ERROR_INVALID_ARGUMENT.code,
                                "vector control length must equal the matrix order",
                                value,
                            ));
                        }
                    }
                }
                EigsOption::Symmetric | EigsOption::Cholesky | EigsOption::Definite => {}
            }
        }
    }
    // A dynamic struct/name can request drop or enable Display.
    if !known_options {
        k = None;
        display = None;
    }
    let vector = factor_output(&matrix, matrix_shape(k, Some(1)), false, true, true);
    let eigenvectors = factor_output(&matrix, matrix_shape(n, k), false, true, true);
    let diagonal = factor_output(&matrix, matrix_shape(k, k), false, true, true);
    // Spectrum domain can change even for real input; callback precision is
    // discovered from an actual successful application, never guessed here.
    let first = match request.outputs.requested.known_count() {
        Some(0 | 1) => vector,
        Some(_) => eigenvectors,
        None => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut result = finish_fixed_outputs(
        entry,
        request,
        vec![first, diagonal, double_scalar()],
        diagnostics,
    );
    if matches!(
        &matrix.kind,
        ValueKindFact::Numeric(_) | ValueKindFact::Logical
    ) {
        result.effects.0.remove(&EffectKind::Unknown);
        if display == Some(false)
            && request
                .outputs
                .requested
                .known_count()
                .is_some_and(|n| n >= 3)
        {
            result.effects.0.remove(&EffectKind::HostCallback);
        }
    }
    result
}
