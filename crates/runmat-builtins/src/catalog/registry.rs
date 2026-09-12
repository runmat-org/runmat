use super::{
    aliases::extend_aliases, entries::extend_catalog_entries, BuiltinCatalogAlias,
    BuiltinCatalogEntry,
};
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

static CATALOG_ALIASES: LazyLock<Vec<&'static BuiltinCatalogAlias>> = LazyLock::new(|| {
    let mut aliases = Vec::new();
    extend_aliases(&mut aliases);
    aliases
});

pub fn builtin_catalog_entries() -> &'static [&'static BuiltinCatalogEntry] {
    CATALOG_ENTRIES.as_slice()
}

pub fn builtin_catalog_aliases() -> &'static [&'static BuiltinCatalogAlias] {
    CATALOG_ALIASES.as_slice()
}

pub fn builtin_catalog_primary_entry_by_name(name: &str) -> Option<&'static BuiltinCatalogEntry> {
    CATALOG_ENTRIES
        .iter()
        .copied()
        .find(|entry| entry.identity.name.eq_ignore_ascii_case(name))
}

pub fn builtin_catalog_alias_by_name(name: &str) -> Option<&'static BuiltinCatalogAlias> {
    CATALOG_ALIASES
        .iter()
        .copied()
        .find(|entry| entry.alias.name.eq_ignore_ascii_case(name))
}

pub fn builtin_catalog_entry_by_name(name: &str) -> Option<&'static BuiltinCatalogEntry> {
    builtin_catalog_primary_entry_by_name(name).or_else(|| {
        builtin_catalog_alias_by_name(name)
            .and_then(|alias| builtin_catalog_primary_entry_by_name(alias.canonical.name))
    })
}

pub fn canonical_builtin_name(name: &str) -> Option<&'static str> {
    builtin_catalog_entry_by_name(name).map(|entry| entry.identity.name)
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
