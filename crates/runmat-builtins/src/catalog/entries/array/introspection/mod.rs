mod iscolumn;
mod isempty;
mod ismatrix;
mod isrow;
mod isscalar;
mod isvector;
mod length;
mod ndims;
mod support;

pub use iscolumn::*;
pub use isempty::*;
pub use ismatrix::*;
pub use isrow::*;
pub use isscalar::*;
pub use isvector::*;
pub use length::*;
pub use ndims::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    iscolumn::ENTRIES,
    isempty::ENTRIES,
    ismatrix::ENTRIES,
    isrow::ENTRIES,
    isscalar::ENTRIES,
    isvector::ENTRIES,
    length::ENTRIES,
    ndims::ENTRIES,
];
