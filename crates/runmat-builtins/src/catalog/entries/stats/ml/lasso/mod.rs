//! Lasso and elastic-net linear regression public contract.
mod contract;
mod documentation;
mod entry;
mod inference;
mod options;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::LASSO_CATALOG_ENTRY;
pub use options::{LassoOption, LassoOptionRole};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LASSO_CATALOG_ENTRY];
