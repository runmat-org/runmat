//! Empirical distribution and CDF plotting identities.
mod cdfplot;
mod contract;
mod ecdf;
mod inference;
mod options;
#[cfg(test)]
mod tests;

use crate::BuiltinCatalogEntry;

pub use cdfplot::CDFPLOT_CATALOG_ENTRY;
pub use contract::*;
pub use ecdf::ECDF_CATALOG_ENTRY;
pub use options::{EcdfBounds, EcdfFunction, EcdfOptionName, EmpiricalBuiltin};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&ECDF_CATALOG_ENTRY, &CDFPLOT_CATALOG_ENTRY];
