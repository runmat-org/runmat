//! TCP socket construction and stream I/O identities.

mod datatype;
mod readline;

pub use datatype::NetDataType;
pub use readline::*;

pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&READLINE_CATALOG_ENTRY];
