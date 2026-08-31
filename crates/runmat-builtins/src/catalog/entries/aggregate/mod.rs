mod structure;

pub use structure::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = &[structure::ENTRIES];
