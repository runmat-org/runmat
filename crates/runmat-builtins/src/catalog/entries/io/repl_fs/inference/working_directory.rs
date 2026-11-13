use crate::{BuiltinCatalogEntry, WorkingDirectoryInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: WorkingDirectoryInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        WorkingDirectoryInferenceRule::Change => super::super::cd::inference::infer(request, entry),
        WorkingDirectoryInferenceRule::Current => {
            super::super::pwd::inference::infer(request, entry)
        }
    }
}
