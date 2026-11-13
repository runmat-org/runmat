mod descriptor;
mod documentation;
mod entry;

pub use descriptor::*;
pub use entry::FFTN_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FFTN_CATALOG_ENTRY];
