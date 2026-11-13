//! Catalog identities of the `fea.SpectrumEvidence` typed policy class.

use crate::*;

mod bounded;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&bounded::FEA_SPECTRUM_EVIDENCE_BOUNDED_CATALOG_ENTRY];
