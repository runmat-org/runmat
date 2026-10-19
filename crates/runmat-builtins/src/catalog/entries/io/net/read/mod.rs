mod contract;
mod entry;
mod documentation;
mod inference;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&READ_CATALOG_ENTRY];
#[cfg(test)]
mod tests;

pub use contract::*;
pub use entry::READ_CATALOG_ENTRY;
