//! Array-introspection builtins.

mod dimension_metadata;
pub(crate) mod height;
pub(crate) mod iscolumn;
pub(crate) mod isempty;
pub(crate) mod ismatrix;
pub(crate) mod isrow;
pub(crate) mod isscalar;
pub(crate) mod isvector;
pub(crate) mod length;
pub(crate) mod ndims;
pub(crate) mod numel;
mod shape_predicate;
mod shape_query;
mod shape_scalar_query;
pub(crate) mod size;
mod structural_output;
pub(crate) mod width;
