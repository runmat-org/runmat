mod descriptor;
mod documentation;
mod entry;

pub use descriptor::*;
pub use entry::FFTSHIFT_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FFTSHIFT_CATALOG_ENTRY];
