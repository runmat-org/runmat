mod accumarray;

pub use accumarray::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&ACCUMARRAY_CATALOG_ENTRY];
