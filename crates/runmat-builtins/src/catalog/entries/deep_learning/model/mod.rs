//! Executable network construction and forward evaluation contracts.

mod dlnetwork;
mod forward;

pub use dlnetwork::{DLNETWORK_CATALOG_ENTRY, DLNETWORK_DESCRIPTOR, DLNETWORK_INVALID_INPUT};
pub use forward::{
    FORWARD_CATALOG_ENTRY, FORWARD_DESCRIPTOR, FORWARD_INTEGER_EXTENSION, FORWARD_INVALID_INPUT,
};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&DLNETWORK_CATALOG_ENTRY, &FORWARD_CATALOG_ENTRY];
