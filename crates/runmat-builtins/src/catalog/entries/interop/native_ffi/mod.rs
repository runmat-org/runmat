//! Canonical native shared-library identities. The runtime foreign adapter
//! remains the sole authority for ABI handles and prepared artifacts.

mod contract;
mod build_interface;
mod calllib;
mod descriptor;
mod documentation_support;
mod inference_support;
mod libfunctions;
mod libisloaded;
mod libpointer;
mod libstruct;
mod loadlibrary;
mod operation;
mod setdatatype;
#[cfg(test)]
mod tests;
mod unloadlibrary;

pub use build_interface::{
    CLIBGEN_BUILD_INTERFACE_CATALOG_ENTRY, CLIBGEN_BUILD_INTERFACE_DESCRIPTOR,
};
pub use calllib::{CALLLIB_CATALOG_ENTRY, CALLLIB_DESCRIPTOR};
pub use contract::{NativeBuildOption, NativeLoadOption};
pub use descriptor::{
    NATIVE_FFI_ERROR_HOST_UNAVAILABLE, NATIVE_FFI_ERROR_INVALID, NATIVE_FFI_ERROR_UNAVAILABLE,
};
pub use libfunctions::{LIBFUNCTIONS_CATALOG_ENTRY, LIBFUNCTIONS_DESCRIPTOR};
pub use libisloaded::{LIBISLOADED_CATALOG_ENTRY, LIBISLOADED_DESCRIPTOR};
pub use libpointer::{LIBPOINTER_CATALOG_ENTRY, LIBPOINTER_DESCRIPTOR};
pub use libstruct::{LIBSTRUCT_CATALOG_ENTRY, LIBSTRUCT_DESCRIPTOR};
pub use loadlibrary::{LOADLIBRARY_CATALOG_ENTRY, LOADLIBRARY_DESCRIPTOR};
pub use operation::NativeFfiOperation;
pub use setdatatype::{SETDATATYPE_CATALOG_ENTRY, SETDATATYPE_DESCRIPTOR};
pub use unloadlibrary::{UNLOADLIBRARY_CATALOG_ENTRY, UNLOADLIBRARY_DESCRIPTOR};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CLIBGEN_BUILD_INTERFACE_CATALOG_ENTRY,
    &LOADLIBRARY_CATALOG_ENTRY,
    &UNLOADLIBRARY_CATALOG_ENTRY,
    &LIBISLOADED_CATALOG_ENTRY,
    &LIBFUNCTIONS_CATALOG_ENTRY,
    &CALLLIB_CATALOG_ENTRY,
    &LIBPOINTER_CATALOG_ENTRY,
    &LIBSTRUCT_CATALOG_ENTRY,
    &SETDATATYPE_CATALOG_ENTRY,
];
