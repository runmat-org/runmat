mod contract;
mod documentation;
mod entry;
mod inference;

pub use contract::GETVAROPTS_DESCRIPTOR;
pub use entry::GETVAROPTS_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&GETVAROPTS_CATALOG_ENTRY];
