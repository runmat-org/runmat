mod facts;
mod fileparts;
mod filesep;
mod fullfile;
mod inference;
mod pathsep;
mod validation;

use crate::{BuiltinCatalogEntry, PathSyntaxInferenceRule};
use runmat_types::{CallInference, CallRequest};

pub use fileparts::*;
pub use filesep::*;
pub use fullfile::*;
pub use pathsep::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &FILEPARTS_CATALOG_ENTRY,
    &FILESEP_CATALOG_ENTRY,
    &FULLFILE_CATALOG_ENTRY,
    &PATHSEP_CATALOG_ENTRY,
];

pub(super) fn infer(
    rule: PathSyntaxInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

#[cfg(test)]
mod tests;
