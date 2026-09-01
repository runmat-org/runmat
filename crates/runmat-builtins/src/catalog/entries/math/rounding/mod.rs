mod ceil;
mod fix;
mod floor;

pub use ceil::*;
pub use fix::*;
pub use floor::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CEIL_CATALOG_ENTRY,
    &FIX_CATALOG_ENTRY,
    &FLOOR_CATALOG_ENTRY,
];
