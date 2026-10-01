//! Mean/median absolute deviation and standardized sample moments.
mod inference;
mod policy;
mod mad;
mod skewness;
mod kurtosis;

pub use kurtosis::*;
pub use mad::*;
pub use policy::{moment_flag_from_f64, MomentControl};
pub use skewness::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &MAD_CATALOG_ENTRY,
    &SKEWNESS_CATALOG_ENTRY,
    &KURTOSIS_CATALOG_ENTRY,
];
