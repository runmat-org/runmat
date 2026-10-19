mod contract;
mod signatures;
mod inference;
mod maxk;
mod mink;
pub use contract::*;
pub use maxk::*;
pub use mink::*;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] =
    &[&MAXK_CATALOG_ENTRY, &MINK_CATALOG_ENTRY];
