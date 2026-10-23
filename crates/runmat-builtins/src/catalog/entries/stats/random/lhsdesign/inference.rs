use super::super::random_inference::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 2, None);
    let shape = ShapeFact::from(vec![literal_count(request, 0), literal_count(request, 1)]);
    finish(
        entry,
        request,
        vec![numeric(
            shape,
            Some(NumericClass::Double),
            ResidencyFact::Host,
        )],
        diagnostics,
        false,
    )
}
