mod descriptor;
mod documentation;
mod entry;

pub use descriptor::*;
pub use entry::IFFTN_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&IFFTN_CATALOG_ENTRY];
