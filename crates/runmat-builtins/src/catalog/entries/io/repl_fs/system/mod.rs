mod contract;
mod documentation;
mod entry;
mod inference;

pub use contract::{SYSTEM_DESCRIPTOR, SYSTEM_INTEGER_AUDIT};
pub use entry::SYSTEM_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SYSTEM_CATALOG_ENTRY];
