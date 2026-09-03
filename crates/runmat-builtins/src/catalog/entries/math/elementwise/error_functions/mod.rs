pub mod erf;
pub mod erfcinv;

pub use erf::*;
pub use erfcinv::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&erf::ERF_CATALOG_ENTRY, &erfcinv::ERFCINV_CATALOG_ENTRY];
