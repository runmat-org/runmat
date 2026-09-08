mod contract;
mod documentation;
mod entry;
mod inference;
mod shorthand;

#[cfg(test)]
mod inference_tests;

pub use contract::*;
pub use entry::CELLFUN_CATALOG_ENTRY;
pub(in crate::catalog) use inference::infer;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CELLFUN_CATALOG_ENTRY];
