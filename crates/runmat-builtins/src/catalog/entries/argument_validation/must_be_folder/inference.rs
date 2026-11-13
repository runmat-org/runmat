use super::super::inference::{definitely_non_text, finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request
        .arguments
        .first()
        .is_some_and(|value| definitely_non_text(value, false))
    {
        reject(&mut diagnostics, 0, "expected text paths");
    }

    // Unknown content never proves predicate success or failure. The public result
    // sequence is nevertheless exactly empty; shared framing handles output requests.
    finish(request, entry, vec![], diagnostics)
}
