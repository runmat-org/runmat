mod documentation;
mod entry;

pub use entry::CATALOG_ENTRY;
pub use CATALOG_ENTRY as ISPREF_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CATALOG_ENTRY];
