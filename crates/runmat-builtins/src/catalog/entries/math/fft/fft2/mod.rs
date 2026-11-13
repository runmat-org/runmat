mod descriptor;
mod documentation;
mod entry;

pub use descriptor::*;
pub use entry::FFT2_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FFT2_CATALOG_ENTRY];
