use crate::{BuiltinCatalogEntry, IoInferenceRule};
use runmat_types::{CallInference, CallRequest};

mod console;
mod repl_fs;

pub(in crate::catalog::inference) fn infer(
    rule: IoInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoInferenceRule::Console(rule) => console::infer(rule, request, entry),
        IoInferenceRule::ReplFs(rule) => repl_fs::infer(rule, request, entry),
    }
}
