mod contract;
mod documentation;
mod entry;
mod inference;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&XLSWRITE_CATALOG_ENTRY];
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::*;
