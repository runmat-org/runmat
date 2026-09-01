mod asin;
mod cos;
mod cosd;
mod cospi;
mod documentation;
mod sin;
mod sind;
mod sinpi;
mod tan;
mod tand;

pub use asin::*;
pub use cos::*;
pub use cosd::*;
pub use cospi::*;
pub use sin::*;
pub use sind::*;
pub use sinpi::*;
pub use tan::*;
pub use tand::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &ASIN_CATALOG_ENTRY,
    &COS_CATALOG_ENTRY,
    &COSD_CATALOG_ENTRY,
    &COSPI_CATALOG_ENTRY,
    &SIN_CATALOG_ENTRY,
    &SINPI_CATALOG_ENTRY,
    &SIND_CATALOG_ENTRY,
    &TAN_CATALOG_ENTRY,
    &TAND_CATALOG_ENTRY,
];
