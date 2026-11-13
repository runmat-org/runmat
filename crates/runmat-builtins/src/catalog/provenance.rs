use serde::Serialize;

/// Physical source provenance emitted by `#[runtime_builtin]`.
///
/// This is development evidence derived from the same declaration that creates
/// the executable registration. It is not a builtin contract or dispatch
/// authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinImplementationProvenance {
    pub name: &'static str,
    pub binding_variant: Option<&'static str>,
    pub source_file: &'static str,
    pub module_path: &'static str,
    pub function: &'static str,
    pub builtin_path: &'static str,
    pub authority: BuiltinImplementationAuthority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinImplementationAuthority {
    CanonicalBinding,
    LegacyFunction,
}

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(BuiltinImplementationProvenance);

#[cfg(not(target_arch = "wasm32"))]
pub fn builtin_implementation_provenance() -> Vec<&'static BuiltinImplementationProvenance> {
    #[cfg(test)]
    INVENTORY_TRAVERSALS.with(|count| count.set(count.get() + 1));
    inventory::iter::<BuiltinImplementationProvenance>
        .into_iter()
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub fn builtin_implementation_provenance() -> Vec<&'static BuiltinImplementationProvenance> {
    crate::wasm_registry::builtin_implementation_provenance()
}

#[cfg(not(target_arch = "wasm32"))]
static NAME_INDEX: std::sync::LazyLock<Vec<&'static BuiltinImplementationProvenance>> =
    std::sync::LazyLock::new(|| {
        let mut entries = builtin_implementation_provenance();
        entries.sort_by_key(|entry| entry.name);
        entries
    });

#[cfg(target_arch = "wasm32")]
thread_local! {
    static NAME_INDEX: std::cell::RefCell<Vec<&'static BuiltinImplementationProvenance>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn index_submitted_provenance(entry: &'static BuiltinImplementationProvenance) {
    NAME_INDEX.with_borrow_mut(|entries| {
        insert(entries, entry);
    });
}

/// Borrow the duplicate-preserving physical evidence for one exact identity.
/// This is an index over registration evidence, not an implementation resolver.
pub fn with_builtin_implementation_provenance_by_name<R>(
    name: &str,
    visit: impl FnOnce(&[&'static BuiltinImplementationProvenance]) -> R,
) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    {
        visit(matching(&NAME_INDEX, name))
    }
    #[cfg(target_arch = "wasm32")]
    {
        NAME_INDEX.with_borrow(|entries| visit(matching(entries, name)))
    }
}

fn matching<'a>(
    entries: &'a [&'static BuiltinImplementationProvenance],
    name: &str,
) -> &'a [&'static BuiltinImplementationProvenance] {
    let first = entries.partition_point(|entry| entry.name < name);
    let count = entries[first..].partition_point(|entry| entry.name == name);
    &entries[first..first + count]
}

#[cfg(any(test, target_arch = "wasm32"))]
fn insert(
    entries: &mut Vec<&'static BuiltinImplementationProvenance>,
    entry: &'static BuiltinImplementationProvenance,
) {
    let at = entries.partition_point(|candidate| candidate.name <= entry.name);
    entries.insert(at, entry);
}

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static INVENTORY_TRAVERSALS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    static PROVENANCE: BuiltinImplementationProvenance = BuiltinImplementationProvenance {
        name: "IndexWitness",
        binding_variant: Some("default"),
        source_file: "test",
        module_path: "test",
        function: "test",
        builtin_path: "test",
        authority: BuiltinImplementationAuthority::CanonicalBinding,
    };

    #[test]
    fn late_provenance_registration_preserves_duplicates_and_case() {
        let mut index = Vec::new();
        assert!(matching(&index, PROVENANCE.name).is_empty());
        insert(&mut index, &PROVENANCE);
        assert_eq!(matching(&index, PROVENANCE.name).len(), 1);
        insert(&mut index, &PROVENANCE);
        assert_eq!(matching(&index, PROVENANCE.name).len(), 2);
        assert!(matching(&index, "indexwitness").is_empty());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn warm_provenance_lookup_does_not_recollect_inventory() {
        with_builtin_implementation_provenance_by_name("missing-index-witness", |_| ());
        let before = INVENTORY_TRAVERSALS.with(std::cell::Cell::get);
        for _ in 0..64 {
            with_builtin_implementation_provenance_by_name("missing-index-witness", |entries| {
                assert!(entries.is_empty());
            });
        }
        assert_eq!(INVENTORY_TRAVERSALS.with(std::cell::Cell::get), before);
    }
}
