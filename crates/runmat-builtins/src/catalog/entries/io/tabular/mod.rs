//! Typed catalog authority for tabular file I/O.

mod csvread;
mod csvwrite;

pub use csvread::*;
pub use csvwrite::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&CSVREAD_CATALOG_ENTRY, &CSVWRITE_CATALOG_ENTRY];
