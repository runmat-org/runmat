mod descriptor;
mod documentation;
mod entry;
mod inference;

pub use descriptor::*;
pub use entry::LSQNONLIN_CATALOG_ENTRY;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&LSQNONLIN_CATALOG_ENTRY];
