mod descriptor;
mod documentation;
mod entry;
mod inference;

pub use descriptor::*;
pub use entry::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&BLKDIAG_CATALOG_ENTRY];
