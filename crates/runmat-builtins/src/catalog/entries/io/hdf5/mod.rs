//! Canonical catalog authority for native HDF5 I/O.

mod common;
mod inference;
pub use common::{
    ERR_ARGUMENT as HDF5_ERROR_ARGUMENT, ERR_IO as HDF5_ERROR_IO,
    ERR_PROVIDER as HDF5_ERROR_PROVIDER, ERR_UNSUPPORTED as HDF5_ERROR_UNSUPPORTED,
    ERR_WASM as HDF5_ERROR_WASM,
};
pub mod h5disp;
pub mod h5info;
pub mod h5read;
pub mod h5write;
pub mod h5writeatt;
pub mod hdf5info;
pub mod hdf5read;
pub mod hdf5write;

pub use h5disp::{H5DISP_CAPTURE_EXTENSION, H5DISP_CATALOG_ENTRY, H5DISP_DESCRIPTOR};
pub use h5info::{H5INFO_CATALOG_ENTRY, H5INFO_DESCRIPTOR};
pub use h5read::{H5READ_CATALOG_ENTRY, H5READ_DESCRIPTOR};
pub use h5write::{H5WRITE_CATALOG_ENTRY, H5WRITE_DESCRIPTOR};
pub use h5writeatt::{H5WRITEATT_CATALOG_ENTRY, H5WRITEATT_DESCRIPTOR};
pub use hdf5info::{HDF5INFO_CATALOG_ENTRY, HDF5INFO_DESCRIPTOR};
pub use hdf5read::{HDF5READ_CATALOG_ENTRY, HDF5READ_DESCRIPTOR};
pub use hdf5write::{HDF5WRITE_CATALOG_ENTRY, HDF5WRITE_DESCRIPTOR};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &h5disp::H5DISP_CATALOG_ENTRY,
    &h5info::H5INFO_CATALOG_ENTRY,
    &h5read::H5READ_CATALOG_ENTRY,
    &h5write::H5WRITE_CATALOG_ENTRY,
    &h5writeatt::H5WRITEATT_CATALOG_ENTRY,
    &hdf5info::HDF5INFO_CATALOG_ENTRY,
    &hdf5read::HDF5READ_CATALOG_ENTRY,
    &hdf5write::HDF5WRITE_CATALOG_ENTRY,
];
