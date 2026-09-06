use crate::{BuiltinCatalogEntry, DirectoryInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: DirectoryInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        DirectoryInferenceRule::Lifecycle(rule) => {
            super::super::directory_lifecycle::infer(rule, request, entry)
        }
        DirectoryInferenceRule::Listing(rule) => {
            super::super::directory_listing::infer(rule, request, entry)
        }
    }
}
