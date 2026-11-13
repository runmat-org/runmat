//! Statistical graphics identities. Each entry owns its public contract.
mod reference_lines;
mod binscatter;
mod qqplot;
mod scatterhist;
mod boxplot;

use crate::BuiltinCatalogEntry;

pub use binscatter::*;
pub use boxplot::*;
pub use qqplot::*;
pub use reference_lines::*;
pub use scatterhist::*;

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &BINSCATTER_CATALOG_ENTRY,
    &LSLINE_CATALOG_ENTRY,
    &QQPLOT_CATALOG_ENTRY,
    &REFLINE_CATALOG_ENTRY,
    &SCATTERHIST_CATALOG_ENTRY,
    &BOXPLOT_CATALOG_ENTRY,
];
