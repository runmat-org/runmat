mod descriptor;
mod documentation;
mod entry;
mod inference;

pub use descriptor::*;
pub use entry::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&REPELEM_CATALOG_ENTRY];
