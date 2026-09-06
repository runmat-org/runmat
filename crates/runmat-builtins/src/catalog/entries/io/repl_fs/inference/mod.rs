mod directory;
mod file;
mod path;
mod working_directory;

use crate::{BuiltinCatalogEntry, IoReplFsInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::entries::io) fn infer(
    rule: IoReplFsInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoReplFsInferenceRule::WorkingDirectory(rule) => {
            working_directory::infer(rule, request, entry)
        }
        IoReplFsInferenceRule::Directory(rule) => directory::infer(rule, request, entry),
        IoReplFsInferenceRule::Environment(rule) => super::environment::infer(rule, request, entry),
        IoReplFsInferenceRule::File(rule) => file::infer(rule, request, entry),
        IoReplFsInferenceRule::Path(rule) => path::infer(rule, request, entry),
        IoReplFsInferenceRule::SourceInventory(rule) => {
            super::source_inventory::infer(rule, request, entry)
        }
    }
}
