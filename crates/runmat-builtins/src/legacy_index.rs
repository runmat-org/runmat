//! A derived, duplicate-preserving view of the existing legacy registrations.
use crate::BuiltinFunction;

#[cfg(not(target_arch = "wasm32"))]
static INDEX: std::sync::LazyLock<Vec<&'static BuiltinFunction>> = std::sync::LazyLock::new(|| {
    let mut entries = crate::builtin_functions();
    entries.sort_by_key(|entry| entry.name);
    entries
});

/// Borrow every legacy registration with this exact dispatch spelling.
/// Unlike the compatibility single-result lookup, duplicates are retained.
pub fn with_builtin_functions_by_exact_name<R>(
    name: &str,
    visit: impl FnOnce(&[&'static BuiltinFunction]) -> R,
) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    {
        visit(matching(&INDEX, name))
    }
    #[cfg(target_arch = "wasm32")]
    {
        crate::wasm_registry::with_function_index(|entries| visit(matching(entries, name)))
    }
}

fn matching<'a>(
    entries: &'a [&'static BuiltinFunction],
    name: &str,
) -> &'a [&'static BuiltinFunction] {
    let first = entries.partition_point(|entry| entry.name < name);
    let count = entries[first..].partition_point(|entry| entry.name == name);
    &entries[first..first + count]
}

#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) fn insert(entries: &mut Vec<&'static BuiltinFunction>, entry: &'static BuiltinFunction) {
    let at = entries.partition_point(|candidate| candidate.name <= entry.name);
    entries.insert(at, entry);
}

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    pub(crate) static INVENTORY_TRAVERSALS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    static FUNCTION: std::sync::LazyLock<BuiltinFunction> = std::sync::LazyLock::new(|| {
        BuiltinFunction::new(
            "IndexWitness",
            "",
            "",
            "",
            "",
            vec![],
            crate::Type::Num,
            None,
            |_| Box::pin(async { Ok(runmat_value::Value::Num(1.0)) }),
            &[],
            false,
            false,
        )
    });

    #[test]
    fn late_legacy_registration_preserves_exact_case_and_duplicates() {
        let mut index = Vec::new();
        assert!(matching(&index, "IndexWitness").is_empty());
        insert(&mut index, &FUNCTION);
        assert_eq!(matching(&index, "IndexWitness").len(), 1);
        assert!(matching(&index, "indexwitness").is_empty());
        insert(&mut index, &FUNCTION);
        assert_eq!(matching(&index, "IndexWitness").len(), 2);
        assert!(std::ptr::eq(
            matching(&index, "IndexWitness")[0],
            &*FUNCTION
        ));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn warm_legacy_lookup_does_not_recollect_inventory() {
        // The builtins-only test binary may contain no runtime registrations.
        // This still verifies repeated misses do not recollect the inventory.
        with_builtin_functions_by_exact_name("missing-index-witness", |_| ());
        let before = INVENTORY_TRAVERSALS.with(std::cell::Cell::get);
        for _ in 0..64 {
            with_builtin_functions_by_exact_name("missing-index-witness", |entries| {
                assert!(entries.is_empty());
            });
        }
        assert_eq!(INVENTORY_TRAVERSALS.with(std::cell::Cell::get), before);
    }
}
