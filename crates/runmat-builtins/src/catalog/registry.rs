use super::{
    definitions::{FULL_CATALOG_ENTRY, ZEROS_CATALOG_ENTRY},
    BuiltinCatalogEntry,
};

static CATALOG_ENTRIES: [&BuiltinCatalogEntry; 2] = [&FULL_CATALOG_ENTRY, &ZEROS_CATALOG_ENTRY];

pub fn builtin_catalog_entries() -> &'static [&'static BuiltinCatalogEntry] {
    &CATALOG_ENTRIES
}

pub fn builtin_catalog_entry_by_name(name: &str) -> Option<&'static BuiltinCatalogEntry> {
    CATALOG_ENTRIES
        .iter()
        .copied()
        .find(|entry| entry.identity.name.eq_ignore_ascii_case(name))
}
