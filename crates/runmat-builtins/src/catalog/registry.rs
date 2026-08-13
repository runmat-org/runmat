use super::{
    definitions::{
        ABS_CATALOG_ENTRY, FULL_CATALOG_ENTRY, GATHER_CATALOG_ENTRY, STRUCT_CATALOG_ENTRY,
        ZEROS_CATALOG_ENTRY,
    },
    BuiltinCatalogEntry,
};

static CATALOG_ENTRIES: [&BuiltinCatalogEntry; 5] = [
    &ABS_CATALOG_ENTRY,
    &FULL_CATALOG_ENTRY,
    &GATHER_CATALOG_ENTRY,
    &STRUCT_CATALOG_ENTRY,
    &ZEROS_CATALOG_ENTRY,
];

pub fn builtin_catalog_entries() -> &'static [&'static BuiltinCatalogEntry] {
    &CATALOG_ENTRIES
}

pub fn builtin_catalog_entry_by_name(name: &str) -> Option<&'static BuiltinCatalogEntry> {
    CATALOG_ENTRIES
        .iter()
        .copied()
        .find(|entry| entry.identity.name.eq_ignore_ascii_case(name))
}
