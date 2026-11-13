use crate::catalog::inference::{argument_error, finish_fixed_outputs, known_invalid_text_scalar};
use crate::*;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 2 {
        diagnostics.push(argument_error(
            "RM-MAKE_ANON-ARITY",
            "make_anon requires 2 arguments",
            request.arguments.len(),
        ));
    }
    for (index, value) in request.arguments.iter().take(2).enumerate() {
        if !matches!(
            value.kind,
            ValueKindFact::Unknown
                | ValueKindFact::String
                | ValueKindFact::Character
                | ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
        ) || (matches!(value.kind, ValueKindFact::String | ValueKindFact::Character)
            && known_invalid_text_scalar(value))
            || (matches!(
                value.kind,
                ValueKindFact::Numeric(_) | ValueKindFact::Logical
            ) && matches!(value.shape.element_count(),Some(count) if count != 1))
            || matches!(
                value.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Complex,
                    ..
                })
            )
            || matches!(
                value.residency,
                ResidencyFact::Device { .. } | ResidencyFact::Remote { .. }
            )
        {
            diagnostics.push(argument_error(
                "RM-MAKE_ANON-TEXT",
                "expected host scalar text or legacy scalar integer/double/logical formatting",
                index,
            ));
        }
    }
    finish_fixed_outputs(
        entry,
        request,
        vec![ValueFact::scalar(ValueKindFact::String)],
        diagnostics,
    )
}
