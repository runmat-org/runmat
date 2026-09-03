mod documentation;
mod function_dispatch;

pub use function_dispatch::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    super::extend_groups(entries, &[function_dispatch::ENTRIES]);
}
