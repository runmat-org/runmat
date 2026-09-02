mod height;
mod iscolumn;
mod isempty;
mod ismatrix;
mod isrow;
mod isscalar;
mod isvector;
mod length;
mod ndims;
mod numel;
mod size;
mod support;
mod width;

pub use height::*;
pub use iscolumn::*;
pub use isempty::*;
pub use ismatrix::*;
pub use isrow::*;
pub use isscalar::*;
pub use isvector::*;
pub use length::*;
pub use ndims::*;
pub use numel::*;
pub use size::*;
pub use width::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[
    height::ENTRIES,
    iscolumn::ENTRIES,
    isempty::ENTRIES,
    ismatrix::ENTRIES,
    isrow::ENTRIES,
    isscalar::ENTRIES,
    isvector::ENTRIES,
    length::ENTRIES,
    ndims::ENTRIES,
    numel::ENTRIES,
    size::ENTRIES,
    width::ENTRIES,
];
