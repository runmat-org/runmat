use crate::catalog::entries::math::linalg::factor::inference::{
    extent, factor_output, matrix_shape, schur_operand,
};
use crate::catalog::inference::{argument_error, finish_fixed_outputs};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, LiteralValue, NumericDomain, ValueFact,
    ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > 2 {
        diagnostics.push(argument_error(
            "RM.SCHUR.INVALID_ARGUMENT",
            "schur expects A and optional mode",
            0,
        ));
    }
    let input = request
        .arguments
        .first()
        .cloned()
        .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue));
    if !request.arguments.is_empty() {
        diagnostics.extend(schur_operand(&input, 0, "RM.SCHUR.INVALID_INPUT"));
    }
    let shape = matrix_shape(extent(&input), extent(&input));
    let mut factor = factor_output(&input, shape, false, false, true);
    if let ValueKindFact::Numeric(numeric) = &input.kind {
        factor.kind = ValueKindFact::Numeric(*numeric);
    }
    let complex_mode = request
        .literals
        .literal_args
        .get(1)
        .map(|literal| match literal {
            LiteralValue::Character(value)
            | LiteralValue::String(value)
            | LiteralValue::Keyword(value) => {
                if value.trim().eq_ignore_ascii_case("complex") {
                    Some(true)
                } else if value.trim().eq_ignore_ascii_case("real") {
                    Some(false)
                } else {
                    diagnostics.push(argument_error(
                        "RM.SCHUR.INVALID_ARGUMENT",
                        "mode must be real or complex",
                        1,
                    ));
                    None
                }
            }
            _ => None,
        });
    if matches!(complex_mode, Some(Some(true))) {
        if let ValueKindFact::Numeric(mut numeric) = factor.kind.clone() {
            numeric.domain = NumericDomain::Complex;
            factor.kind = ValueKindFact::Numeric(numeric);
        }
    } else if request.arguments.len() == 2
        && !matches!(complex_mode, Some(Some(false)))
        && !matches!(&input.kind, ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Complex)
    {
        // A runtime mode value may request complex Schur factors for a real A.
        // The output class/shape remain known, but its domain does not.
        factor.kind = ValueKindFact::Unknown;
    }
    let first = match request.outputs.requested.known_count() {
        Some(1) => factor.clone(),
        Some(2) => factor.clone(),
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    finish_fixed_outputs(entry, request, vec![first, factor], diagnostics)
}
