//! Catalog identities of the `fea.ModalTruncation` typed policy class.

use crate::*;

mod new_method;
#[cfg(test)]
mod tests;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] =
    &[&new_method::FEA_MODAL_TRUNCATION_NEW_CATALOG_ENTRY];
