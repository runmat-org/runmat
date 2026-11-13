//! Canonical contracts for legacy formatted and delimited text imports.

pub mod importdata;
pub mod textscan;

pub use importdata::*;
pub use textscan::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &importdata::IMPORTDATA_CATALOG_ENTRY,
    &textscan::TEXTSCAN_CATALOG_ENTRY,
];
