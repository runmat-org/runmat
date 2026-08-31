use super::{definitions::DOMAIN_ENTRY_GROUPS, BuiltinCatalogEntry};
use std::sync::LazyLock;

/// Canonical entries composed from domain-owned definition groups.
///
/// Each contract family registers its entries beside their definitions. The
/// root registry composes domain groups only, so adding a builtin never
/// requires editing a second, repository-wide list.
static CATALOG_ENTRIES: LazyLock<Vec<&'static BuiltinCatalogEntry>> = LazyLock::new(|| {
    DOMAIN_ENTRY_GROUPS
        .iter()
        .flat_map(|families| families.iter())
        .flat_map(|entries| entries.iter().copied())
        .collect()
});

pub fn builtin_catalog_entries() -> &'static [&'static BuiltinCatalogEntry] {
    CATALOG_ENTRIES.as_slice()
}

pub fn builtin_catalog_entry_by_name(name: &str) -> Option<&'static BuiltinCatalogEntry> {
    CATALOG_ENTRIES
        .iter()
        .copied()
        .find(|entry| entry.identity.name.eq_ignore_ascii_case(name))
}
