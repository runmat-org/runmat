//! Catalog identities of the `fea.ExecutionStep` typed policy class.

use crate::*;

mod new_method;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&new_method::FEA_EXECUTION_STEP_NEW_CATALOG_ENTRY];
