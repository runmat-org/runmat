mod facts;
mod getenv;
mod inference;
mod isenv;
mod setenv;
mod unsetenv;
mod validation;

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

pub use getenv::*;
pub use isenv::*;
pub use setenv::*;
pub use unsetenv::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &GETENV_CATALOG_ENTRY,
    &SETENV_CATALOG_ENTRY,
    &ISENV_CATALOG_ENTRY,
    &UNSETENV_CATALOG_ENTRY,
];

pub(super) fn infer(
    rule: crate::EnvironmentInferenceRule,
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    inference::infer(rule, request, entry)
}

#[cfg(test)]
mod tests;
