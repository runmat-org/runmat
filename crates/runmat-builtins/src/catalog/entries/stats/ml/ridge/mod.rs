//! Ridge-regression contract and documentation.
mod contract;
mod documentation;
mod entry;
mod inference;
mod integer;
mod scaling;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::RIDGE_CATALOG_ENTRY;
pub use integer::*;
pub use scaling::RidgeScaling;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&RIDGE_CATALOG_ENTRY];
