mod creation;
mod documentation;

pub use creation::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[creation::ENTRIES];
