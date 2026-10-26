//! Catalog identities of the `fea.AccuracyCriterion` typed policy class.

use crate::*;

mod new_method;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&new_method::FEA_ACCURACY_CRITERION_NEW_CATALOG_ENTRY];
