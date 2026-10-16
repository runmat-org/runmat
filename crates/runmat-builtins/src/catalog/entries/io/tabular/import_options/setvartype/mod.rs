mod contract;
mod documentation;
mod entry;
mod inference;

pub use contract::SETVARTYPE_DESCRIPTOR;
pub use entry::SETVARTYPE_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SETVARTYPE_CATALOG_ENTRY];
