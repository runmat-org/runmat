mod contract;
mod control;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;
pub use contract::*;
pub(super) use control::scalar_text_input;
pub use control::*;
pub use entry::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&PARQUETREAD_CATALOG_ENTRY];
