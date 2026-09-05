mod log;
mod log10;
mod log1p;
mod log2;

pub use log::*;
pub use log10::*;
pub use log1p::*;
pub use log2::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &LOG_CATALOG_ENTRY,
    &LOG10_CATALOG_ENTRY,
    &LOG1P_CATALOG_ENTRY,
    &LOG2_CATALOG_ENTRY,
];
