use super::{entries::extend_catalog_entries, BuiltinCatalogEntry};
use std::sync::LazyLock;

/// Canonical entries composed from domain-owned entry groups.
///
/// Each contract family registers its entries beside their typed contracts. The
/// root registry composes domain groups only, so adding a builtin never
/// requires editing a second, repository-wide list.
static CATALOG_ENTRIES: LazyLock<Vec<&'static BuiltinCatalogEntry>> = LazyLock::new(|| {
    let mut entries = Vec::new();
    extend_catalog_entries(&mut entries);
    entries
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

/// Resolves a class-qualified builtin method from typed class and member
/// identities. Callers do not synthesize callable names or infer class
/// identity from text.
pub fn builtin_catalog_entry_for_class_method(
    class: &runmat_types::ClassIdentity,
    method: &runmat_types::MethodName,
) -> Option<&'static BuiltinCatalogEntry> {
    let qualified = format!("{}.{}", class.display_name(), method.0);
    builtin_catalog_entry_by_name(&qualified)
}
