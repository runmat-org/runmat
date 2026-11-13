//! Loss functions for deep-learning arrays.

mod crossentropy;

pub use crossentropy::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&crossentropy::CROSSENTROPY_CATALOG_ENTRY];
