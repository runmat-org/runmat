mod console;

pub use console::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, &[console::ENTRIES]);
}
