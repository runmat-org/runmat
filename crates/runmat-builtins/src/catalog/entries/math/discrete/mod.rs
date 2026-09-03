mod factorial;

pub use factorial::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&FACTORIAL_CATALOG_ENTRY];
