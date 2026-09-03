mod operators;
mod relational;
mod tests;

pub use operators::*;
pub use relational::*;

pub use tests::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, tests::ENTRY_GROUPS);
    super::extend_groups(entries, operators::ENTRY_GROUPS);
    super::extend_groups(entries, relational::ENTRY_GROUPS);
}
