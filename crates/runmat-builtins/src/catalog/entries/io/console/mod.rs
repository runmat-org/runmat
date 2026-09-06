mod clc;

use crate::{BuiltinCatalogEntry, IoConsoleInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use clc::*;

pub(super) fn infer(
    rule: IoConsoleInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoConsoleInferenceRule::ClearConsole => clc::inference::infer(request, entry),
    }
}

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = clc::ENTRIES;
