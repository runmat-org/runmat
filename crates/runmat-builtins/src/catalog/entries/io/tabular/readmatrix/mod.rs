mod contract;
mod control;
mod documentation;
mod entry;
mod inference;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&READMATRIX_CATALOG_ENTRY];
#[cfg(test)]
mod tests;

pub use contract::*;
pub use control::*;
pub use entry::*;
