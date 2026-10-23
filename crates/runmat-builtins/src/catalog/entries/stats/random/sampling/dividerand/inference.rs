use super::super::super::random_inference::*;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = arity(request, 1, Some(4));
    let row = numeric(
        ShapeFact::from(vec![Some(1), None]),
        Some(NumericClass::Double),
        ResidencyFact::Host,
    );
    finish(
        entry,
        request,
        vec![row.clone(), row.clone(), row],
        diagnostics,
        false,
    )
}
