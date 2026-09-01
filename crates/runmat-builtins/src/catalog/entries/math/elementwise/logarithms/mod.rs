mod documentation;
mod log1p;
mod log2;
mod natural_and_common;

pub use log1p::*;
pub use log2::*;
pub use natural_and_common::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &LOG_CATALOG_ENTRY,
    &LOG10_CATALOG_ENTRY,
    &LOG1P_CATALOG_ENTRY,
    &LOG2_CATALOG_ENTRY,
];
