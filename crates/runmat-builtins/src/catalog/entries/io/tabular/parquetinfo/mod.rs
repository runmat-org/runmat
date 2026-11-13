mod contract;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;
pub use contract::*;
pub use entry::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PARQUETINFO_CATALOG_ENTRY];
