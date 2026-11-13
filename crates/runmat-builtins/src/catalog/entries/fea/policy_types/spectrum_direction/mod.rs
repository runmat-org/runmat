//! Catalog identities of the `fea.SpectrumDirection` typed policy class.

use crate::*;

mod new_method;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&new_method::FEA_SPECTRUM_DIRECTION_NEW_CATALOG_ENTRY];
