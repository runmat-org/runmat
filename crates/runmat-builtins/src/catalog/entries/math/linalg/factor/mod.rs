mod decomposition;

pub use decomposition::*;

pub(super) fn extend_entries(entries: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    entries.extend(decomposition::ENTRIES.iter().copied());
}
