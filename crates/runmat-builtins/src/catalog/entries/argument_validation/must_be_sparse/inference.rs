use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if input.storage != StorageFact::Unknown
            && input.storage != StorageFact::Sparse
            && input.shape.element_count().is_some_and(|n| n != 0)
        {
            reject(&mut diagnostics, 0, "expected sparse or empty");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
