mod deg2rad;
mod rad2deg;

pub use deg2rad::*;
pub use rad2deg::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&DEG2RAD_CATALOG_ENTRY, &RAD2DEG_CATALOG_ENTRY];
