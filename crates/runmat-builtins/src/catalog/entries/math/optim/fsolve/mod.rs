mod descriptor;
mod documentation;
mod entry;
mod inference;

pub use descriptor::*;
pub use entry::FSOLVE_CATALOG_ENTRY;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FSOLVE_CATALOG_ENTRY];
