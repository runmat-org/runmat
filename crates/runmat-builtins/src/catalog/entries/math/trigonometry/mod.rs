mod cos;
mod documentation;
mod sin;
mod sinpi;
mod tan;

pub use cos::*;
pub use sin::*;
pub use sinpi::*;
pub use tan::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &COS_CATALOG_ENTRY,
    &SIN_CATALOG_ENTRY,
    &SINPI_CATALOG_ENTRY,
    &TAN_CATALOG_ENTRY,
];
