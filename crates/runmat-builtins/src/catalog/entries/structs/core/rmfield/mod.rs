mod contract;
mod documentation;
mod entry;
mod inference;

#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::RMFIELD_CATALOG_ENTRY;
pub(in crate::catalog) use inference::infer;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&RMFIELD_CATALOG_ENTRY];
