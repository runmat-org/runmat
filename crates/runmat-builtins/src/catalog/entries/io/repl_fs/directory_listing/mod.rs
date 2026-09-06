mod dir;
mod facts;
mod inference;
mod ls;
mod validation;

use crate::{BuiltinCatalogEntry, DirectoryListingInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use dir::*;
pub use ls::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&DIR_CATALOG_ENTRY, &LS_CATALOG_ENTRY];

pub(super) fn infer(
    rule: DirectoryListingInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

#[cfg(test)]
mod tests;
