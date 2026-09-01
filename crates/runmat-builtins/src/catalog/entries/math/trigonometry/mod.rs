mod cos;
mod cospi;
mod documentation;
mod sin;
mod sind;
mod sinpi;
mod tan;

pub use cos::*;
pub use cospi::*;
pub use sin::*;
pub use sind::*;
pub use sinpi::*;
pub use tan::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &COS_CATALOG_ENTRY,
    &COSPI_CATALOG_ENTRY,
    &SIN_CATALOG_ENTRY,
    &SINPI_CATALOG_ENTRY,
    &SIND_CATALOG_ENTRY,
    &TAN_CATALOG_ENTRY,
];
