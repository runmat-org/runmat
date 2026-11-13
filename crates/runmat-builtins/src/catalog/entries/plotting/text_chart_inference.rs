//! Chart handles are scalar doubles; coordinate/selector checks remain at Runtime.
use crate::catalog::inference::{argument_error, finish_fixed};
use crate::BuiltinCatalogEntry;
use runmat_types::*;

#[cfg(test)]
mod tests;

pub(super) fn infer(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    minimum_inputs: usize,
) -> CallInference {
    let diagnostics = if request.arguments.len() < minimum_inputs {
        vec![argument_error(
            "RM-CATALOG-TEXT-CHART-ARITY",
            "insufficient chart inputs",
            request.arguments.len(),
        )]
    } else {
        Vec::new()
    };
    let output = if diagnostics.is_empty() {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Scalar,
            StorageFact::Dense,
        )
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    finish_fixed(entry, request, output, diagnostics)
}
