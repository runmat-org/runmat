//! Typed catalog authority for tabular file I/O.

mod csvread;
mod csvwrite;
mod dlmread;
mod dlmwrite;

pub use csvread::*;
pub use csvwrite::*;
pub use dlmread::*;
pub use dlmwrite::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CSVREAD_CATALOG_ENTRY,
    &CSVWRITE_CATALOG_ENTRY,
    &DLMREAD_CATALOG_ENTRY,
    &DLMWRITE_CATALOG_ENTRY,
];
