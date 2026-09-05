use crate::{BuiltinCatalogEntry, IoInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer(
    rule: IoInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoInferenceRule::ClearConsole => super::super::io::clear_console(request, entry),
        IoInferenceRule::CurrentDirectory => super::super::io::current_directory(request, entry),
    }
}
