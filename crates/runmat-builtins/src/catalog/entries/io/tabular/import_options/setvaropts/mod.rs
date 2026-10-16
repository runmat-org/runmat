mod contract;
mod documentation;
mod entry;
mod inference;

pub use contract::SETVAROPTS_DESCRIPTOR;
pub use entry::SETVAROPTS_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SETVAROPTS_CATALOG_ENTRY];
