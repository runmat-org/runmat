use crate::{BuiltinCatalogEntry, FileInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: FileInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        FileInferenceRule::Transfer(rule) => {
            super::super::file_transfer::infer(rule, request, entry)
        }
    }
}
