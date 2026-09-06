use crate::{BuiltinCatalogEntry, IoReplFsInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io) fn infer(
    rule: IoReplFsInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoReplFsInferenceRule::ChangeDirectory => super::cd::inference::infer(request, entry),
        IoReplFsInferenceRule::CurrentDirectory => super::pwd::inference::infer(request, entry),
        IoReplFsInferenceRule::DirectoryLifecycle(rule) => {
            super::directory_lifecycle::infer(rule, request, entry)
        }
        IoReplFsInferenceRule::Environment(rule) => super::environment::infer(rule, request, entry),
        IoReplFsInferenceRule::PathSyntax(rule) => super::path_syntax::infer(rule, request, entry),
        IoReplFsInferenceRule::SearchPath(rule) => super::search_path::infer(rule, request, entry),
        IoReplFsInferenceRule::TemporaryPath(rule) => {
            super::temporary_path::infer(rule, request, entry)
        }
    }
}
