use super::super::inference::finish;
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let diagnostics = Vec::new();
    finish(
        request,
        entry,
        vec![ValueFact::scalar(ValueKindFact::Logical)],
        diagnostics,
    )
}
