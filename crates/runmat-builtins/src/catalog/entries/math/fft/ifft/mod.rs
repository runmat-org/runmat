mod descriptor;
mod documentation;
mod entry;

pub use descriptor::*;
pub use entry::IFFT_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&IFFT_CATALOG_ENTRY];
