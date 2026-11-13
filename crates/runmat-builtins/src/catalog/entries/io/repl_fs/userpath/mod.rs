mod contract;
mod control;
mod documentation;
mod entry;
mod inference;
pub(super) const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[&USERPATH_CATALOG_ENTRY];

pub use contract::{USERPATH_DESCRIPTOR, USERPATH_INTEGER_AUDIT, USERPATH_PARTIAL_PATH_EXTENSION};
pub use control::{UserPathControl, UserPathControlError, UserPathForm};
pub use entry::USERPATH_CATALOG_ENTRY;
