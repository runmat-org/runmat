use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = vec![];
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-AXIS-ARITY",
            "axis accepts at most one argument",
            1,
        ));
    }
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.residency = ResidencyFact::Host;
    if request.arguments.is_empty() {
        output = limit_vector();
    } else if let Some(argument) = request.arguments.first() {
        let invalid_kind = match argument.kind {
            ValueKindFact::Unknown => false,
            ValueKindFact::Character => {
                !crate::catalog::inference::possible_character_row(&argument.shape)
            }
            ValueKindFact::String | ValueKindFact::Logical => {
                !matches!(argument.storage, StorageFact::Unknown | StorageFact::Scalar)
            }
            ValueKindFact::Numeric(numeric) => {
                numeric.domain == NumericDomain::Complex
                    || argument.storage == StorageFact::Sparse
                    || argument
                        .shape
                        .element_count()
                        .is_some_and(|n| !matches!(n, 1 | 4 | 6 | 8))
            }
            _ => true,
        };
        if invalid_kind
            || matches!(
                argument.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-AXIS-INPUT",
                "axis requires host real limits, a scalar handle/visibility control, or a mode",
                0,
            ));
        }
        if let Some(text) = request.literals.literal_string_at(0) {
            if !matches!(
                text.trim().to_ascii_lowercase().as_str(),
                "equal" | "auto" | "tight" | "image" | "on" | "off" | "manual" | "ij" | "xy"
            ) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-AXIS-MODE",
                    "unsupported axis mode",
                    0,
                ));
            }
            output = ValueFact::scalar(ValueKindFact::Logical);
        } else if argument.kind == ValueKindFact::Logical
            || (argument.numeric().is_some()
                && argument
                    .shape
                    .element_count()
                    .is_some_and(|n| matches!(n, 4 | 6 | 8)))
        {
            output = ValueFact::scalar(ValueKindFact::Logical);
        }
        if let Some(values) = request.literals.literal_vector_at(0) {
            if matches!(values.len(), 4 | 6 | 8) {
                for pair in values.chunks_exact(2) {
                    let lower = LiteralContext::numeric_from_literal(&pair[0]);
                    let upper = LiteralContext::numeric_from_literal(&pair[1]);
                    let invalid = lower.is_some_and(|n| n.is_nan() || n == f64::INFINITY)
                        || upper.is_some_and(|n| n.is_nan() || n == f64::NEG_INFINITY)
                        || lower
                            .zip(upper)
                            .is_some_and(|(lo, hi)| lo.is_finite() && hi.is_finite() && hi <= lo);
                    if invalid {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-AXIS-LIMITS",
                            "axis limits must be ordered with supported automatic endpoints",
                            0,
                        ));
                        break;
                    }
                }
            }
        }
        // A numeric scalar may be a live axes identity. Neither existence nor
        // whether the query has four or six limits is a static numeric fact.
    }
    finish_fixed_outputs(entry, request, vec![output], diagnostics)
}

fn limit_vector() -> ValueFact {
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), None]),
        StorageFact::Dense,
    )
}
