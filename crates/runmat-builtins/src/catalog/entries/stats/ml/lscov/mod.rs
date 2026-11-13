//! Weighted and generalized least-squares identity.
mod algorithm;
mod contract;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use algorithm::LscovAlgorithm;
pub use contract::*;
pub use entry::LSCOV_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LSCOV_CATALOG_ENTRY];
