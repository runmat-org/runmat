use crate::{
    catalog::inference::{argument_error, finish_fixed_outputs},
    BuiltinCatalogEntry,
};
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-CLEAR_ALL_MEMOIZED_CACHES-ARITY",
            "unsupported number of input arguments",
            request.arguments.len(),
        ));
    }
    let outputs = vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))];
    finish_fixed_outputs(entry, request, outputs, diagnostics)
}
