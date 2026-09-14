use super::{
    aliases::extend_aliases,
    entries::{extend_constants, extend_entries},
    BuiltinCatalogAlias, BuiltinCatalogEntry, BuiltinConstantCatalogEntry,
};
use std::cmp::Ordering;
use std::sync::LazyLock;

/// Canonical entries composed from domain-owned entry groups.
///
/// Each contract family registers its entries beside their typed contracts. The
/// root registry composes domain groups only, so adding a builtin never
/// requires editing a second, repository-wide list.
static CATALOG_ENTRIES: LazyLock<Vec<&'static BuiltinCatalogEntry>> = LazyLock::new(|| {
    let mut entries = Vec::new();
    extend_entries(&mut entries);
    entries
});

static CATALOG_ALIASES: LazyLock<Vec<&'static BuiltinCatalogAlias>> = LazyLock::new(|| {
    let mut aliases = Vec::new();
    extend_aliases(&mut aliases);
    aliases
});

static CATALOG_CONSTANTS: LazyLock<Vec<BuiltinConstantCatalogEntry>> = LazyLock::new(|| {
    let mut constants = Vec::new();
    extend_constants(&mut constants);
    constants.sort_unstable_by_key(|constant| constant.name);
    constants
});

static CATALOG_ENTRY_NAME_INDEX: LazyLock<Vec<&'static BuiltinCatalogEntry>> =
    LazyLock::new(|| {
        let mut entries = CATALOG_ENTRIES.iter().copied().collect::<Vec<_>>();
        entries.sort_unstable_by(|left, right| {
            compare_ascii_case_insensitive(left.identity.name, right.identity.name)
        });
        entries
    });

static CATALOG_ALIAS_NAME_INDEX: LazyLock<Vec<&'static BuiltinCatalogAlias>> =
    LazyLock::new(|| {
        let mut aliases = CATALOG_ALIASES.iter().copied().collect::<Vec<_>>();
        aliases.sort_unstable_by(|left, right| {
            compare_ascii_case_insensitive(left.alias.name, right.alias.name)
        });
        aliases
    });

pub fn builtin_catalog_entries() -> &'static [&'static BuiltinCatalogEntry] {
    CATALOG_ENTRIES.as_slice()
}

pub fn builtin_catalog_aliases() -> &'static [&'static BuiltinCatalogAlias] {
    CATALOG_ALIASES.as_slice()
}

pub fn builtin_constant_catalog_entries() -> &'static [BuiltinConstantCatalogEntry] {
    CATALOG_CONSTANTS.as_slice()
}

pub fn builtin_constant_catalog_entry_by_name(
    name: &str,
) -> Option<&'static BuiltinConstantCatalogEntry> {
    CATALOG_CONSTANTS
        .binary_search_by(|entry| entry.name.cmp(name))
        .ok()
        .map(|index| &CATALOG_CONSTANTS[index])
}

pub fn builtin_catalog_primary_entry_by_name(name: &str) -> Option<&'static BuiltinCatalogEntry> {
    CATALOG_ENTRY_NAME_INDEX
        .binary_search_by(|entry| compare_ascii_case_insensitive(entry.identity.name, name))
        .ok()
        .map(|index| CATALOG_ENTRY_NAME_INDEX[index])
}

pub fn builtin_catalog_alias_by_name(name: &str) -> Option<&'static BuiltinCatalogAlias> {
    CATALOG_ALIAS_NAME_INDEX
        .binary_search_by(|entry| compare_ascii_case_insensitive(entry.alias.name, name))
        .ok()
        .map(|index| CATALOG_ALIAS_NAME_INDEX[index])
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

fn compare_ascii_case_insensitive(left: &str, right: &str) -> Ordering {
    left.bytes()
        .map(|byte| byte.to_ascii_lowercase())
        .cmp(right.bytes().map(|byte| byte.to_ascii_lowercase()))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuiltinConstantKind;

    #[test]
    fn primary_name_index_is_case_insensitive_without_changing_canonical_spelling() {
        let entry = builtin_catalog_entries()
            .first()
            .copied()
            .expect("catalog must not be empty");
        let requested = entry.identity.name.to_ascii_uppercase();
        let resolved = builtin_catalog_primary_entry_by_name(&requested).expect("indexed entry");
        assert_eq!(resolved.identity.name, entry.identity.name);
        assert!(builtin_catalog_primary_entry_by_name("not_a_runmat_builtin").is_none());
    }

    #[test]
    fn case_insensitive_comparator_preserves_qualified_segment_order() {
        assert_eq!(
            compare_ascii_case_insensitive("DataArray.read", "dataarray.READ"),
            Ordering::Equal,
        );
        assert_eq!(
            compare_ascii_case_insensitive("dataarray.read", "dataarray.write"),
            Ordering::Less,
        );
    }

    #[test]
    fn constant_aggregation_preserves_typed_identity_and_fact_authority() {
        let entries = builtin_constant_catalog_entries();
        assert!(
            !entries.is_empty(),
            "the constant catalog must not be empty"
        );
        let names = entries
            .iter()
            .map(|entry| entry.name)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), entries.len());

        let mut represented_kinds = [false; 3];
        for entry in entries {
            assert!(!entry.name.is_empty());
            assert_eq!(
                builtin_constant_catalog_entry_by_name(entry.name),
                Some(entry),
                "the exact declared identity must resolve to its typed catalog entry"
            );
            match entry.kind {
                BuiltinConstantKind::RealDouble => {
                    represented_kinds[0] = true;
                    assert!(matches!(
                        entry.fact().kind,
                        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
                            class: runmat_types::NumericClass::Double,
                            domain: runmat_types::NumericDomain::Real,
                        })
                    ));
                }
                BuiltinConstantKind::ComplexDouble => {
                    represented_kinds[1] = true;
                    assert!(matches!(
                        entry.fact().kind,
                        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
                            class: runmat_types::NumericClass::Double,
                            domain: runmat_types::NumericDomain::Complex,
                        })
                    ));
                }
                BuiltinConstantKind::Logical => {
                    represented_kinds[2] = true;
                    assert_eq!(entry.fact().kind, runmat_types::ValueKindFact::Logical)
                }
            }
        }
        assert!(
            represented_kinds.into_iter().all(std::convert::identity),
            "the catalog must exercise every typed constant projection"
        );
        assert!(builtin_constant_catalog_entry_by_name("not_a_runmat_constant").is_none());
    }

    #[test]
    fn constant_provenance_matches_its_declaring_catalog_module() {
        fn source_module_suffix(source_file: &str) -> String {
            let normalized = source_file.replace('\\', "/");
            let (_, catalog_relative) = normalized
                .rsplit_once("/src/catalog/")
                .or_else(|| normalized.split_once("src/catalog/"))
                .unwrap_or_else(|| {
                    panic!("catalog provenance is outside src/catalog: {source_file}")
                });
            let module_relative = catalog_relative
                .strip_suffix("/mod.rs")
                .or_else(|| catalog_relative.strip_suffix(".rs"))
                .unwrap_or_else(|| panic!("catalog provenance is not Rust source: {source_file}"));
            format!("catalog::{}", module_relative.replace('/', "::"))
        }

        let mut source_by_module = std::collections::BTreeMap::new();
        let mut module_by_source = std::collections::BTreeMap::new();
        for entry in builtin_constant_catalog_entries() {
            let provenance = entry.provenance;
            assert!(!provenance.source_file.is_empty(), "{}", entry.name);
            assert!(!provenance.module_path.is_empty(), "{}", entry.name);
            assert!(
                provenance
                    .module_path
                    .ends_with(&source_module_suffix(provenance.source_file)),
                "constant {} reports source {} but module {}",
                entry.name,
                provenance.source_file,
                provenance.module_path
            );
            if let Some(previous) =
                source_by_module.insert(provenance.module_path, provenance.source_file)
            {
                assert_eq!(previous, provenance.source_file);
            }
            if let Some(previous) =
                module_by_source.insert(provenance.source_file, provenance.module_path)
            {
                assert_eq!(previous, provenance.module_path);
            }
        }
    }
}
