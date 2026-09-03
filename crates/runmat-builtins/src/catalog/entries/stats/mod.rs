mod random;

pub use random::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, random::ENTRY_GROUPS);
}
