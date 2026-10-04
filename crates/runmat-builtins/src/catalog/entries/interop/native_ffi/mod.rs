//! Canonical native shared-library identities. The runtime foreign adapter
//! remains the sole authority for ABI handles and prepared artifacts.

mod contract;
mod descriptor;
mod documentation_support;
mod inference_support;
mod loadlibrary;
mod unloadlibrary;

pub use descriptor::{
    LOADLIBRARY_DESCRIPTOR, NATIVE_FFI_ERROR_HOST_UNAVAILABLE, NATIVE_FFI_ERROR_INVALID,
    NATIVE_FFI_ERROR_UNAVAILABLE, NATIVE_FFI_INTEGER_AUDIT, UNLOADLIBRARY_DESCRIPTOR,
};
pub use loadlibrary::LOADLIBRARY_CATALOG_ENTRY;
pub use unloadlibrary::UNLOADLIBRARY_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&LOADLIBRARY_CATALOG_ENTRY, &UNLOADLIBRARY_CATALOG_ENTRY];
