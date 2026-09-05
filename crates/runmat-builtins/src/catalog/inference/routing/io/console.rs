use crate::{BuiltinCatalogEntry, IoConsoleInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(super) fn infer(
    rule: IoConsoleInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoConsoleInferenceRule::ClearConsole => {
            super::super::super::io::clear_console(request, entry)
        }
    }
}
