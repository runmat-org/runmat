mod contract;
mod diagnostics;
mod dimensions;
mod documentation;
mod entry;
mod inference;

#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::NUM2CELL_CATALOG_ENTRY;
pub(in crate::catalog) use inference::infer;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&NUM2CELL_CATALOG_ENTRY];
