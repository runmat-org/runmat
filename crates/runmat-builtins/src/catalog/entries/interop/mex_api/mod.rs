//! Native MEX build and target identities. The build adapter owns option,
//! compiler, ABI, and artifact validation; these entries declare public calls.

mod contract;
mod documentation_support;
mod inference_support;
mod mex;
mod mexcuda;
mod mexext;
#[cfg(test)]
mod tests;

pub use contract::{
    MEX_ERROR_BUILD_ARGUMENTS, MEX_ERROR_BUILD_FAILED, MEX_ERROR_OUTPUTS, MEX_ERROR_UNAVAILABLE,
};
pub use mex::{MEX_CATALOG_ENTRY, MEX_DESCRIPTOR};
pub use mexcuda::{MEXCUDA_CATALOG_ENTRY, MEXCUDA_DESCRIPTOR};
pub use mexext::{MEXEXT_CATALOG_ENTRY, MEXEXT_DESCRIPTOR};

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &MEX_CATALOG_ENTRY,
    &MEXCUDA_CATALOG_ENTRY,
    &MEXEXT_CATALOG_ENTRY,
];
