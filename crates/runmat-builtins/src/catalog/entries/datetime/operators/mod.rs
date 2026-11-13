//! Typed datetime method-family authority. Runtime and inference share operation identity, not names.
use crate::*;
mod contract;
mod documentation;
mod inference;
mod operation;
#[cfg(test)]
mod tests;
pub use contract::*;
pub use operation::DatetimeBinaryOperation;
pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &DATETIME_EQ_CATALOG_ENTRY,
    &DATETIME_NE_CATALOG_ENTRY,
    &DATETIME_LT_CATALOG_ENTRY,
    &DATETIME_LE_CATALOG_ENTRY,
    &DATETIME_GT_CATALOG_ENTRY,
    &DATETIME_GE_CATALOG_ENTRY,
    &DATETIME_PLUS_CATALOG_ENTRY,
    &DATETIME_MINUS_CATALOG_ENTRY,
];
