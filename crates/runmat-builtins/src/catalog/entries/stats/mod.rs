mod random;

pub use random::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = random::ENTRY_GROUPS;
