mod console;
mod repl_fs;

pub use console::*;
pub use repl_fs::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, &[console::ENTRIES, repl_fs::ENTRIES]);
}
