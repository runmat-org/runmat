use crate::{BuiltinCatalogEntry, IoInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog) fn infer(
    rule: IoInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoInferenceRule::Console(rule) => super::console::infer(rule, request, entry),
        IoInferenceRule::ReplFs(rule) => super::repl_fs::infer(rule, request, entry),
    }
}
