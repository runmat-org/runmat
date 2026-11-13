mod inference;
mod what;

use crate::{BuiltinCatalogEntry, SourceInventoryInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use what::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&WHAT_CATALOG_ENTRY];

pub(super) fn infer(
    rule: SourceInventoryInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

#[cfg(test)]
mod tests;
