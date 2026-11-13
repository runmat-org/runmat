use super::super::inference::{finish, has_stored_shape, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request
        .arguments
        .first()
        .filter(|input| has_stored_shape(input))
    {
        if input.shape.element_count().is_some_and(|n| n > 1) {
            reject(&mut diagnostics, 0, "expected scalar or empty");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
