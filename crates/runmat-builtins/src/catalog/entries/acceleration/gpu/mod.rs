//! Composition of host-side active-provider metadata queries.

mod device;
mod info;

pub use device::*;
pub use info::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&GPU_DEVICE_CATALOG_ENTRY, &GPU_INFO_CATALOG_ENTRY];
