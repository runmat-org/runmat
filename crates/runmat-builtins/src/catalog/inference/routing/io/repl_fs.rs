use crate::{BuiltinCatalogEntry, IoReplFsInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: IoReplFsInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoReplFsInferenceRule::ChangeDirectory => {
            super::super::super::io::change_directory(request, entry)
        }
        IoReplFsInferenceRule::CurrentDirectory => {
            super::super::super::io::current_directory(request, entry)
        }
        IoReplFsInferenceRule::SearchPath => super::super::super::io::search_path(request, entry),
    }
}
