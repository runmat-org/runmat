mod contract;
mod documentation;
mod entry;
mod inference;
mod signatures;

#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::CELL_CATALOG_ENTRY;
pub(in crate::catalog) use inference::infer;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CELL_CATALOG_ENTRY];
