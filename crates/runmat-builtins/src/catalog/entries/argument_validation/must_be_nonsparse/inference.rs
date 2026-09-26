use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if input.storage == StorageFact::Sparse {
            reject(&mut diagnostics, 0, "expected nonsparse storage");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
