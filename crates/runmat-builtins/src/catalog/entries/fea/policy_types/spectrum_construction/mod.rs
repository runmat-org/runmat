//! Catalog identities of the `fea.SpectrumConstruction` typed policy class.

use crate::*;

mod authored_design;
mod from_acceleration_history_exact;
mod from_acceleration_history_recursive;
#[cfg(test)]
mod tests;

pub(super) const CONSTRUCTION_ROLE: &str = concat!(
    "A construction policy records where a spectrum's ordinates came from. It is ",
    "provenance: `fea.ResponseSpectrum.new` stores the ordinates you pass and does not ",
    "recompute them from the construction.",
);

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &authored_design::FEA_SPECTRUM_CONSTRUCTION_AUTHORED_DESIGN_CATALOG_ENTRY,
    &from_acceleration_history_exact::FEA_SPECTRUM_CONSTRUCTION_FROM_ACCELERATION_HISTORY_EXACT_CATALOG_ENTRY,
    &from_acceleration_history_recursive::FEA_SPECTRUM_CONSTRUCTION_FROM_ACCELERATION_HISTORY_RECURSIVE_CATALOG_ENTRY,
];
