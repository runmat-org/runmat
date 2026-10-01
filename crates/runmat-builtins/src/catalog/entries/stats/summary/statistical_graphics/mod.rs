//! Statistical graphics identities. Each entry owns its public contract.
mod reference_lines;
mod binscatter;
mod qqplot;

use crate::BuiltinCatalogEntry;

pub use binscatter::*;
pub use qqplot::*;
pub use reference_lines::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &BINSCATTER_CATALOG_ENTRY,
    &LSLINE_CATALOG_ENTRY,
    &QQPLOT_CATALOG_ENTRY,
    &REFLINE_CATALOG_ENTRY,
];
