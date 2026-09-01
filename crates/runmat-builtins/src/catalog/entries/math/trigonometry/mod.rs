mod cos;
mod documentation;
mod sin;

pub use cos::*;
pub use sin::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&COS_CATALOG_ENTRY, &SIN_CATALOG_ENTRY];
