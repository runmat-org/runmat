pub mod elementwise;

pub use elementwise::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = elementwise::ENTRY_GROUPS;
