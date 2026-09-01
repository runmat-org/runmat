mod acos;
mod acosh;
mod asin;
mod asinh;
mod atan;
mod atanh;
mod cos;
mod cosd;
mod cosh;
mod cospi;
mod sin;
mod sind;
mod sinh;
mod sinpi;
mod tan;
mod tand;
mod tanh;

pub use acos::*;
pub use acosh::*;
pub use asin::*;
pub use asinh::*;
pub use atan::*;
pub use atanh::*;
pub use cos::*;
pub use cosd::*;
pub use cosh::*;
pub use cospi::*;
pub use sin::*;
pub use sind::*;
pub use sinh::*;
pub use sinpi::*;
pub use tan::*;
pub use tand::*;
pub use tanh::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &ACOS_CATALOG_ENTRY,
    &ACOSH_CATALOG_ENTRY,
    &ASIN_CATALOG_ENTRY,
    &ASINH_CATALOG_ENTRY,
    &ATAN_CATALOG_ENTRY,
    &ATANH_CATALOG_ENTRY,
    &COS_CATALOG_ENTRY,
    &COSH_CATALOG_ENTRY,
    &COSD_CATALOG_ENTRY,
    &COSPI_CATALOG_ENTRY,
    &SIN_CATALOG_ENTRY,
    &SINPI_CATALOG_ENTRY,
    &SINH_CATALOG_ENTRY,
    &SIND_CATALOG_ENTRY,
    &TAN_CATALOG_ENTRY,
    &TAND_CATALOG_ENTRY,
    &TANH_CATALOG_ENTRY,
];
