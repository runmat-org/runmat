//! Catalog identities of the `fea.SpectrumDamping` typed policy class.

use crate::*;

mod quality_factor;
mod ratio;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &ratio::FEA_SPECTRUM_DAMPING_RATIO_CATALOG_ENTRY,
    &quality_factor::FEA_SPECTRUM_DAMPING_QUALITY_FACTOR_CATALOG_ENTRY,
];
