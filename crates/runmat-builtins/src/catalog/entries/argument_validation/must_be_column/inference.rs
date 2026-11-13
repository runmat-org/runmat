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
        if input.shape.known_dims().is_some_and(|dims| {
            dims.get(1).is_some_and(|d| d.is_some_and(|n| n != 1))
                || dims.iter().skip(2).any(|d| d.is_some_and(|n| n != 1))
        }) {
            reject(&mut diagnostics, 0, "expected a column shape");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
