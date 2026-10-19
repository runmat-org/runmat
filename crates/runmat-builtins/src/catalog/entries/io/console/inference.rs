//! Routing for the retained console family inference rule.

use crate::{BuiltinCatalogEntry, IoConsoleInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io) fn infer(
    rule: IoConsoleInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoConsoleInferenceRule::ClearConsole => super::clc::inference::infer(request, entry),
    }
}
