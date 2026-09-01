mod documentation;
mod sin;

pub use sin::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&SIN_CATALOG_ENTRY];
