use crate::{BuiltinCatalogEntry, TemporaryPathInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: TemporaryPathInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        TemporaryPathInferenceRule::Directory => super::tempdir::inference::infer(request, entry),
        TemporaryPathInferenceRule::UniqueName => super::tempname::inference::infer(request, entry),
    }
}
