use super::super::inference::{definitely_non_text, finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if definitely_non_text(input, true) {
            reject(&mut diagnostics, 0, "expected supported text");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
