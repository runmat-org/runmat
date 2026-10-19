//! Ordinary multiple linear regression identity.
mod contract;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::REGRESS_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&REGRESS_CATALOG_ENTRY];
