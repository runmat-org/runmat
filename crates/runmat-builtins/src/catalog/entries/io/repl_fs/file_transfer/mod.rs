mod copyfile;
mod facts;
mod inference;
mod movefile;
mod validation;

use crate::{BuiltinCatalogEntry, FileTransferInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use copyfile::*;
pub use movefile::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&COPYFILE_CATALOG_ENTRY, &MOVEFILE_CATALOG_ENTRY];

pub(super) fn infer(
    rule: FileTransferInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

#[cfg(test)]
mod tests;
