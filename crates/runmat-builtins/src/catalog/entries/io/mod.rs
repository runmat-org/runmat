mod console;
mod repl_fs;

use crate::{BuiltinCatalogEntry, IoInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use console::*;
pub use repl_fs::*;

pub(in crate::catalog) fn infer(
    rule: IoInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoInferenceRule::Console(rule) => console::infer(rule, request, entry),
        IoInferenceRule::ReplFs(rule) => repl_fs::infer(rule, request, entry),
    }
}

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, &[console::ENTRIES]);
    repl_fs::extend_entries(entries);
}
