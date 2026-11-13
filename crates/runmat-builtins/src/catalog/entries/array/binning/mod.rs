mod discretize;

pub use discretize::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&DISCRETIZE_CATALOG_ENTRY];
