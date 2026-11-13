use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    for (index, input) in request.arguments.iter().enumerate() {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(_)
                | ValueKindFact::String
                | ValueKindFact::Character
        ) || input
            .numeric()
            .is_some_and(|n| n.domain == NumericDomain::Complex)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-CLF-TARGET",
                "expected figure numbers, all, or reset",
                index,
            ));
        }
    }
    // Empty, singleton, multiple, and all forms return a scalar handle/count.
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
