mod descriptor;
mod documentation;
mod entry;

pub use descriptor::*;
pub use entry::IFFT2_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&IFFT2_CATALOG_ENTRY];
