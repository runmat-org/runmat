use crate::{
    catalog::inference::{argument_error, finish_fixed, known_invalid_text_scalar},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let first = request.arguments.first();
    let start = if first.is_some_and(|input| matches!(input.kind, ValueKindFact::Numeric(_)))
        || request.literals.literal_string_at(0).is_some_and(|s| {
            s.trim().eq_ignore_ascii_case("next") || s.trim().parse::<u32>().is_ok()
        }) {
        1
    } else {
        0
    };
    // Unknown first values may be either target handles or property names.
    let ambiguous_prefix = first.is_some_and(|input| {
        input.kind == ValueKindFact::Unknown
            || (matches!(input.kind, ValueKindFact::String | ValueKindFact::Character)
                && request.literals.literal_string_at(0).is_none())
    });
    if !ambiguous_prefix {
        if !(request.arguments.len() - start).is_multiple_of(2) {
            diagnostics.push(argument_error(
                "RM-CATALOG-FIGURE-PAIRS",
                "property arguments must be name/value pairs",
                start,
            ));
        }
        for index in (start..request.arguments.len().saturating_sub(1)).step_by(2) {
            if known_invalid_text_scalar(&request.arguments[index]) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-FIGURE-PROPERTY",
                    "property names must be scalar text",
                    index,
                ));
            }
        }
    }
    if start == 1
        && first.is_some_and(|input| !input.is_scalar() && input.shape.element_count().is_some())
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-FIGURE-TARGET",
            "figure target must be scalar",
            0,
        ));
    }
    finish_fixed(
        entry,
        request,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        diagnostics,
    )
}
