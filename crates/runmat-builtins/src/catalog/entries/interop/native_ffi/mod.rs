//! Canonical native shared-library identities. The runtime foreign adapter
//! remains the sole authority for ABI handles and prepared artifacts.

mod contract;
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

pub use calllib::CALLLIB_CATALOG_ENTRY;
pub use contract::NativeLoadOption;
pub use descriptor::{
    CALLLIB_DESCRIPTOR, LIBFUNCTIONS_DESCRIPTOR, LIBISLOADED_DESCRIPTOR, LIBPOINTER_DESCRIPTOR,
    LIBSTRUCT_DESCRIPTOR, LOADLIBRARY_DESCRIPTOR, NATIVE_FFI_ERROR_HOST_UNAVAILABLE,
    NATIVE_FFI_ERROR_INVALID, NATIVE_FFI_ERROR_UNAVAILABLE, NATIVE_FFI_INTEGER_AUDIT,
    SETDATATYPE_DESCRIPTOR, UNLOADLIBRARY_DESCRIPTOR,
};
pub use libfunctions::LIBFUNCTIONS_CATALOG_ENTRY;
pub use libisloaded::LIBISLOADED_CATALOG_ENTRY;
pub use libpointer::LIBPOINTER_CATALOG_ENTRY;
pub use libstruct::LIBSTRUCT_CATALOG_ENTRY;
pub use loadlibrary::LOADLIBRARY_CATALOG_ENTRY;
pub use operation::NativeFfiOperation;
pub use setdatatype::SETDATATYPE_CATALOG_ENTRY;
pub use unloadlibrary::UNLOADLIBRARY_CATALOG_ENTRY;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &LOADLIBRARY_CATALOG_ENTRY,
    &UNLOADLIBRARY_CATALOG_ENTRY,
    &LIBISLOADED_CATALOG_ENTRY,
    &LIBFUNCTIONS_CATALOG_ENTRY,
    &CALLLIB_CATALOG_ENTRY,
    &LIBPOINTER_CATALOG_ENTRY,
    &LIBSTRUCT_CATALOG_ENTRY,
    &SETDATATYPE_CATALOG_ENTRY,
];
