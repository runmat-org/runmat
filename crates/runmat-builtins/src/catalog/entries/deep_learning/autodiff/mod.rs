//! Differentiable dlarray arithmetic and traced gradients.

mod array_methods;
mod dlgradient;

pub use array_methods::*;
pub use dlgradient::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &DLARRAY_MINUS_CATALOG_ENTRY,
    &DLARRAY_MTIMES_CATALOG_ENTRY,
    &DLARRAY_PLUS_CATALOG_ENTRY,
    &DLARRAY_RDIVIDE_CATALOG_ENTRY,
    &DLARRAY_SUM_CATALOG_ENTRY,
    &DLARRAY_TIMES_CATALOG_ENTRY,
    &DLGRADIENT_CATALOG_ENTRY,
];
