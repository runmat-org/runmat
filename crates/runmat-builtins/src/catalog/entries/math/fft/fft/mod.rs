mod descriptor;
mod documentation;
mod entry;

pub use descriptor::*;
pub use entry::FFT_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FFT_CATALOG_ENTRY];
