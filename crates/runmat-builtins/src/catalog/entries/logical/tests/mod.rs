mod isfinite;
mod isgpuarray;
mod isinf;
mod islogical;
mod isnan;
mod isnumeric;
mod isreal;
mod issparse;
mod support;

pub use isfinite::*;
pub use isgpuarray::*;
pub use isinf::*;
pub use islogical::*;
pub use isnan::*;
pub use isnumeric::*;
pub use isreal::*;
pub use issparse::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    isfinite::ENTRIES,
    isgpuarray::ENTRIES,
    isinf::ENTRIES,
    islogical::ENTRIES,
    isnan::ENTRIES,
    isnumeric::ENTRIES,
    isreal::ENTRIES,
    issparse::ENTRIES,
];
