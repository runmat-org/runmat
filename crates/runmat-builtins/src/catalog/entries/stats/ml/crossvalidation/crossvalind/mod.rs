//! Legacy cross-validation index and mask API.

mod contract;
mod documentation;
mod entry;
mod inference;
#[cfg(test)]
mod tests;

pub use contract::{
    CROSSVALIND_DESCRIPTOR, CROSSVALIND_ERROR_INTERNAL, CROSSVALIND_ERROR_INVALID_ARGUMENT,
};
pub use entry::CROSSVALIND_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&CROSSVALIND_CATALOG_ENTRY];
