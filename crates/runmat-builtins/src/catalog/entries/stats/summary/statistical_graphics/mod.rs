//! Statistical graphics identities. Each entry owns its public contract.
mod reference_lines;
mod qqplot;

use crate::BuiltinCatalogEntry;

pub use qqplot::*;
pub use reference_lines::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &LSLINE_CATALOG_ENTRY,
    &QQPLOT_CATALOG_ENTRY,
    &REFLINE_CATALOG_ENTRY,
];
