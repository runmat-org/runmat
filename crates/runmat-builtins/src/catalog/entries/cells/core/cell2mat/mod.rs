mod contract;
mod documentation;
mod entry;
mod examples;
mod inference;

#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::CELL2MAT_CATALOG_ENTRY;
pub(in crate::catalog) use inference::infer;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CELL2MAT_CATALOG_ENTRY];
