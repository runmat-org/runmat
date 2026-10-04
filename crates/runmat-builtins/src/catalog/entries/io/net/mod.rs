//! TCP socket construction and stream I/O identities.

mod readline;

pub use readline::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&READLINE_CATALOG_ENTRY];
