mod facts;
mod inference;
mod mkdir;
mod rmdir;
mod validation;

use crate::{BuiltinCatalogEntry, DirectoryLifecycleInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use mkdir::*;
pub use rmdir::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MKDIR_CATALOG_ENTRY, &RMDIR_CATALOG_ENTRY];

pub(super) fn infer(
    rule: DirectoryLifecycleInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

#[cfg(test)]
mod tests;
