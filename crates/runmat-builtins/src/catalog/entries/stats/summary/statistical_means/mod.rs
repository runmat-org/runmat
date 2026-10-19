//! Canonical contracts for four related, but independently callable, means.
mod inference;
mod policy;
mod geomean;
mod harmmean;
mod rms;
mod rmse;

pub use geomean::*;
pub use harmmean::*;
pub use policy::*;
pub use rms::*;
pub use rmse::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &GEOMEAN_CATALOG_ENTRY,
    &HARMMEAN_CATALOG_ENTRY,
    &RMS_CATALOG_ENTRY,
    &RMSE_CATALOG_ENTRY,
];
