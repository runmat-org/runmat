mod inference;
mod tempdir;
mod tempname;

use crate::{BuiltinCatalogEntry, TemporaryPathInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use tempdir::*;
pub use tempname::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&TEMPDIR_CATALOG_ENTRY, &TEMPNAME_CATALOG_ENTRY];

pub(super) fn infer(
    rule: TemporaryPathInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

#[cfg(test)]
mod tests;
