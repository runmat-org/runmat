//! Catalog identities of the `fea.DirectionalCombination` typed policy class.

use crate::*;

mod percentage;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&percentage::FEA_DIRECTIONAL_COMBINATION_PERCENTAGE_CATALOG_ENTRY];
