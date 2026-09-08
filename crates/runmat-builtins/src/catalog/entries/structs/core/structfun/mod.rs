mod contract;
mod documentation;
mod entry;
mod inference;

#[cfg(test)]
mod inference_tests;

pub use contract::*;
pub use entry::STRUCTFUN_CATALOG_ENTRY;
pub(in crate::catalog) use inference::infer;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&STRUCTFUN_CATALOG_ENTRY];
