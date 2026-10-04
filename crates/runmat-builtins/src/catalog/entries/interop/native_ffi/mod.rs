//! Canonical native shared-library identities. The runtime foreign adapter
//! remains the sole authority for ABI handles and prepared artifacts.

mod contract;
mod descriptor;
mod documentation_support;
mod inference_support;
mod libfunctions;
mod libisloaded;
mod loadlibrary;
mod unloadlibrary;

pub use descriptor::{
    LIBFUNCTIONS_DESCRIPTOR, LIBISLOADED_DESCRIPTOR, LOADLIBRARY_DESCRIPTOR,
    NATIVE_FFI_ERROR_HOST_UNAVAILABLE, NATIVE_FFI_ERROR_INVALID, NATIVE_FFI_ERROR_UNAVAILABLE,
    NATIVE_FFI_INTEGER_AUDIT, UNLOADLIBRARY_DESCRIPTOR,
};
pub use libfunctions::LIBFUNCTIONS_CATALOG_ENTRY;
pub use libisloaded::LIBISLOADED_CATALOG_ENTRY;
pub use loadlibrary::LOADLIBRARY_CATALOG_ENTRY;
pub use unloadlibrary::UNLOADLIBRARY_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &LOADLIBRARY_CATALOG_ENTRY,
    &UNLOADLIBRARY_CATALOG_ENTRY,
    &LIBISLOADED_CATALOG_ENTRY,
    &LIBFUNCTIONS_CATALOG_ENTRY,
];
