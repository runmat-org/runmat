mod relational;
mod tests;

pub use relational::*;

pub use tests::*;

pub(super) const ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] = tests::ENTRY_GROUPS;
pub(super) const RELATIONAL_ENTRY_GROUPS: &[&[&crate::BuiltinCatalogEntry]] =
    relational::ENTRY_GROUPS;
