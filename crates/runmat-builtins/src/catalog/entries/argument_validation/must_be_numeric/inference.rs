use super::super::inference::{finish, reject};
use crate::BuiltinCatalogEntry;
use runmat_types::*;
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if let Some(input) = request.arguments.first() {
        if !matches!(
            input.kind,
            ValueKindFact::Unknown | ValueKindFact::Numeric(_)
        ) {
            reject(&mut diagnostics, 0, "expected numeric data");
        }
    }

    finish(request, entry, vec![], diagnostics)
}
