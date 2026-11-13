mod descriptor;
mod documentation;
mod entry;

pub use descriptor::*;
pub use entry::IFFTSHIFT_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&IFFTSHIFT_CATALOG_ENTRY];
