//! Calendar-duration family topology and catalog aggregation only.

mod contracts;
mod declarations;
mod documentation;
mod inference;
#[cfg(test)]
mod tests;

pub use declarations::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CALDAYS_CATALOG_ENTRY,
    &CALENDARDURATION_CATALOG_ENTRY,
    &CALMONTHS_CATALOG_ENTRY,
    &CALQUARTERS_CATALOG_ENTRY,
    &CALWEEKS_CATALOG_ENTRY,
    &CALYEARS_CATALOG_ENTRY,
    &ISCALENDARDURATION_CATALOG_ENTRY,
];
