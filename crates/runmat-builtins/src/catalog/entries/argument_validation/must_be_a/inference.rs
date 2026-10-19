use super::super::inference::{definitely_non_text, finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request
        .arguments
        .get(1)
        .is_some_and(|names| definitely_non_text(names, false))
    {
        reject(&mut diagnostics, 1, "expected text class names");
    }

    finish(request, entry, vec![], diagnostics)
}
