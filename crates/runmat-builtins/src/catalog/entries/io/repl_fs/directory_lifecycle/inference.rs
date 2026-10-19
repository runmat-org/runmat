use crate::{BuiltinCatalogEntry, DirectoryLifecycleInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: DirectoryLifecycleInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        DirectoryLifecycleInferenceRule::Create => super::mkdir::inference::infer(request, entry),
        DirectoryLifecycleInferenceRule::Remove => super::rmdir::inference::infer(request, entry),
    }
}
