//! Typed catalog authority for tabular file I/O.

mod csvread;
mod csvwrite;
mod dlmread;
mod dlmwrite;
mod readmatrix;
mod writematrix;
mod writecell;
mod xlswrite;
mod xlsread;

pub use csvread::*;
pub use csvwrite::*;
pub use dlmread::*;
pub use dlmwrite::*;
pub use readmatrix::*;
pub use writecell::*;
pub use writematrix::*;
pub use xlsread::*;
pub use xlswrite::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[
    &CSVREAD_CATALOG_ENTRY,
    &CSVWRITE_CATALOG_ENTRY,
    &DLMREAD_CATALOG_ENTRY,
    &DLMWRITE_CATALOG_ENTRY,
    &READMATRIX_CATALOG_ENTRY,
    &WRITEMATRIX_CATALOG_ENTRY,
    &WRITECELL_CATALOG_ENTRY,
    &XLSWRITE_CATALOG_ENTRY,
    &XLSREAD_CATALOG_ENTRY,
];
