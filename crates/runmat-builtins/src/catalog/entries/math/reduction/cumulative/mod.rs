mod controls;
mod inference;
#[cfg(test)]
mod tests;
mod cumsum;
mod cumprod;
mod cummin;
mod cummax;
pub use controls::*;
pub use cummax::*;
pub use cummin::*;
pub use cumprod::*;
pub use cumsum::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CUMSUM_CATALOG_ENTRY,
    &CUMPROD_CATALOG_ENTRY,
    &CUMMIN_CATALOG_ENTRY,
    &CUMMAX_CATALOG_ENTRY,
];
