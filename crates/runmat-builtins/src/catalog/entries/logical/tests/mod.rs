mod allfinite;
mod isfinite;
mod isgpuarray;
mod isinf;
mod islogical;
mod isnan;
mod isnumeric;
mod isreal;
mod issparse;
mod support;

pub use allfinite::*;
pub use isfinite::*;
pub use isgpuarray::*;
pub use isinf::*;
pub use islogical::*;
pub use isnan::*;
pub use isnumeric::*;
pub use isreal::*;
pub use issparse::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    allfinite::ENTRIES,
    isfinite::ENTRIES,
    isgpuarray::ENTRIES,
    isinf::ENTRIES,
    islogical::ENTRIES,
    isnan::ENTRIES,
    isnumeric::ENTRIES,
    isreal::ENTRIES,
    issparse::ENTRIES,
];
