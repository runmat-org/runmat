mod contract;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;
pub use contract::*;
pub use entry::STRUCT_CATALOG_ENTRY;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&STRUCT_CATALOG_ENTRY];
