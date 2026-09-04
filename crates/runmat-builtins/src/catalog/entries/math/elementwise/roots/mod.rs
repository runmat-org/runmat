mod realsqrt;
mod sqrt;

pub use realsqrt::*;
pub use sqrt::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&SQRT_CATALOG_ENTRY, &REALSQRT_CATALOG_ENTRY];
