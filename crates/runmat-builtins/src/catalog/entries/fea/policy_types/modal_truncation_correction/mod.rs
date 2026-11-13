//! Catalog identities of the `fea.ModalTruncationCorrection` typed policy class.

use crate::*;

mod missing_mass;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&missing_mass::FEA_MODAL_TRUNCATION_CORRECTION_MISSING_MASS_CATALOG_ENTRY];
