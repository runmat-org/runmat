mod documentation;
mod structure;

pub use structure::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, &[structure::ENTRIES]);
}
