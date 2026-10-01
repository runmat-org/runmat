//! Cross-validation object and mask access contracts.

mod contract;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::{CVPARTITION_CATALOG_ENTRY, TEST_CATALOG_ENTRY, TRAINING_CATALOG_ENTRY};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CVPARTITION_CATALOG_ENTRY,
    &TEST_CATALOG_ENTRY,
    &TRAINING_CATALOG_ENTRY,
];
