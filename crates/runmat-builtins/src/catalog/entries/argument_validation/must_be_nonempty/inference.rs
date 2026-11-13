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
        if input.shape.element_count() == Some(0) {
            reject(&mut diagnostics, 0, "expected a nonempty value");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
