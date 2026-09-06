mod addpath;
mod cd;
mod genpath;
mod path;
mod pwd;
mod rmpath;
mod search_path;

use crate::{BuiltinCatalogEntry, IoReplFsInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use addpath::*;
pub use cd::*;
pub use genpath::*;
pub use path::*;
pub use pwd::*;
pub use rmpath::*;

pub(super) fn infer(
    rule: IoReplFsInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    match rule {
        IoReplFsInferenceRule::ChangeDirectory => cd::inference::infer(request, entry),
        IoReplFsInferenceRule::CurrentDirectory => pwd::inference::infer(request, entry),
        IoReplFsInferenceRule::SearchPath(rule) => search_path::infer(rule, request, entry),
    }
}

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    addpath::ENTRIES,
    cd::ENTRIES,
    genpath::ENTRIES,
    path::ENTRIES,
    pwd::ENTRIES,
    rmpath::ENTRIES,
];
