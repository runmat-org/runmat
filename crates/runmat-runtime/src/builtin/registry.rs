use super::RuntimeBuiltinBinding;
use std::collections::{BTreeMap, BTreeSet};

pub mod registration;

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(RuntimeBuiltinBinding);

#[cfg(not(target_arch = "wasm32"))]
pub fn runtime_builtin_bindings() -> Vec<RuntimeBuiltinBinding> {
    #[cfg(test)]
    INVENTORY_TRAVERSALS.with(|count| count.set(count.get() + 1));
    inventory::iter::<RuntimeBuiltinBinding>()
        .copied()
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub mod wasm_registry {
    use super::RuntimeBuiltinBinding;
    use std::cell::RefCell;

    thread_local! {
        static BINDINGS: RefCell<Vec<&'static RuntimeBuiltinBinding>> = const { RefCell::new(Vec::new()) };
        static NAME_INDEX: RefCell<super::registration::RegistrationIndex> =
            RefCell::new(super::registration::RegistrationIndex::default());
    }

    pub fn submit(binding: RuntimeBuiltinBinding) {
        NAME_INDEX.with_borrow_mut(|index| {
            index.insert(binding);
        });
        BINDINGS.with_borrow_mut(|bindings| bindings.push(Box::leak(Box::new(binding))));
    }

    pub(super) fn with_index<R>(visit: impl FnOnce(&[RuntimeBuiltinBinding]) -> R) -> R {
        NAME_INDEX.with_borrow(|index| visit(&index.entries))
    }

    pub(super) fn with_index_and_generation<R>(
        visit: impl FnOnce(&[RuntimeBuiltinBinding], u64) -> R,
    ) -> R {
        NAME_INDEX.with_borrow(|index| visit(&index.entries, index.generation))
    }

    pub fn bindings() -> Vec<&'static RuntimeBuiltinBinding> {
        BINDINGS.with_borrow(Clone::clone)
    }
}

#[cfg(target_arch = "wasm32")]
pub fn runtime_builtin_bindings() -> Vec<RuntimeBuiltinBinding> {
    wasm_registry::bindings().into_iter().copied().collect()
}

pub fn runtime_builtin_bindings_by_name(name: &str) -> Vec<RuntimeBuiltinBinding> {
    with_runtime_builtin_bindings_by_name(name, |entries| entries.to_vec())
}

#[cfg(not(target_arch = "wasm32"))]
static NAME_INDEX: std::sync::LazyLock<Vec<RuntimeBuiltinBinding>> =
    std::sync::LazyLock::new(|| {
        let mut entries = runtime_builtin_bindings();
        // Stable sorting preserves dispatcher selection priority among equal names.
        entries.sort_by(|left, right| {
            compare_names(left.identity.builtin.name, right.identity.builtin.name)
        });
        entries
    });

fn compare_names(left: &str, right: &str) -> std::cmp::Ordering {
    left.bytes()
        .map(|byte| byte.to_ascii_lowercase())
        .cmp(right.bytes().map(|byte| byte.to_ascii_lowercase()))
}

/// Borrow all compiled bindings for an executable dispatch spelling, retaining
/// duplicates and their registration order without allocating a result vector.
pub fn with_runtime_builtin_bindings_by_name<R>(
    name: &str,
    visit: impl FnOnce(&[RuntimeBuiltinBinding]) -> R,
) -> R {
    #[cfg(not(target_arch = "wasm32"))]
    {
        visit(matching_bindings(&NAME_INDEX, name))
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_registry::with_index(|entries| visit(matching_bindings(entries, name)))
    }
}

fn matching_bindings<'a>(
    entries: &'a [RuntimeBuiltinBinding],
    name: &str,
) -> &'a [RuntimeBuiltinBinding] {
    &entries[matching_binding_range(entries, name)]
}

fn matching_binding_range(entries: &[RuntimeBuiltinBinding], name: &str) -> std::ops::Range<usize> {
    let first =
        entries.partition_point(|entry| compare_names(entry.identity.builtin.name, name).is_lt());
    let count = entries[first..]
        .partition_point(|entry| entry.identity.builtin.name.eq_ignore_ascii_case(name));
    first..first + count
}

#[cfg(any(test, target_arch = "wasm32"))]
fn insert_binding(entries: &mut Vec<RuntimeBuiltinBinding>, binding: RuntimeBuiltinBinding) {
    let at = entries.partition_point(|entry| {
        !compare_names(entry.identity.builtin.name, binding.identity.builtin.name).is_gt()
    });
    entries.insert(at, binding);
}

#[cfg(all(test, not(target_arch = "wasm32")))]
thread_local! {
    static INVENTORY_TRAVERSALS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeBuiltinRegistryError {
    pub identity: String,
    pub message: String,
}

pub fn validate_runtime_builtin_registry() -> Vec<RuntimeBuiltinRegistryError> {
    let catalog = runmat_builtins::builtin_catalog_entries();
    let actual = runtime_builtin_bindings();
    let mut errors = Vec::new();
    let mut actual_counts = BTreeMap::new();
    for binding in &actual {
        *actual_counts.entry(binding.identity).or_insert(0usize) += 1;
    }
    for (identity, count) in &actual_counts {
        if *count > 1 {
            errors.push(registry_error(
                identity,
                "runtime binding identity is registered more than once",
            ));
        }
    }

    let mut declared = BTreeSet::new();
    for entry in catalog {
        if runmat_builtins::builtin_function_by_name(entry.identity.name).is_some() {
            errors.push(RuntimeBuiltinRegistryError {
                identity: entry.identity.name.to_string(),
                message: "catalog identity also has a legacy BuiltinFunction authority".into(),
            });
        }
        for binding in entry.bindings {
            let identity = entry.binding_identity(binding);
            declared.insert(identity);
            let count = actual_counts.get(&identity).copied().unwrap_or(0);
            if matches!(
                binding.availability,
                runmat_builtins::BuiltinBindingAvailability::Required
            ) && count != 1
            {
                errors.push(registry_error(
                    &identity,
                    "required runtime binding is not registered exactly once",
                ));
            }
        }
    }
    for binding in actual {
        if !declared.contains(&binding.identity) {
            errors.push(registry_error(
                &binding.identity,
                "runtime binding has no canonical catalog declaration",
            ));
        } else if binding.physical_provenance().is_none() {
            errors.push(registry_error(
                &binding.identity,
                "canonical runtime binding has no matching declaration-owned physical provenance",
            ));
        }
    }
    errors
}

fn registry_error(
    identity: &runmat_builtins::BuiltinBindingIdentity,
    message: impl Into<String>,
) -> RuntimeBuiltinRegistryError {
    RuntimeBuiltinRegistryError {
        identity: format!("{}#{}", identity.builtin.name, identity.variant),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_catalog_and_runtime_bindings_join_one_to_one() {
        let errors = validate_runtime_builtin_registry();
        assert!(errors.is_empty(), "{errors:#?}");
        assert_eq!(runtime_builtin_bindings_by_name("full").len(), 1);
        assert!(runmat_builtins::builtin_function_by_name("full").is_none());
    }

    #[test]
    fn max_dispatch_identity_has_exactly_one_canonical_default_binding() {
        let entry = &runmat_builtins::MAX_CATALOG_ENTRY;
        assert_eq!(entry.identity.name, "max");
        assert_eq!(
            entry.bindings,
            runmat_builtins::REQUIRED_DEFAULT_BINDING.as_slice()
        );
        let bindings = runtime_builtin_bindings_by_name(entry.identity.name);
        assert_eq!(bindings.len(), 1);
        assert_eq!(
            bindings[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
        assert!(runmat_builtins::builtin_function_by_name(entry.identity.name).is_none());

        let observed =
            crate::dispatcher::call_builtin(entry.identity.name, &[runmat_value::Value::Num(5.0)])
                .expect("dispatch canonical max");
        assert_eq!(observed, runmat_value::Value::Num(5.0));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn warm_binding_lookup_borrows_without_inventory_traversal() {
        let first = with_runtime_builtin_bindings_by_name("sin", |entries| {
            assert_eq!(entries.len(), 1);
            entries.as_ptr()
        });
        let before = INVENTORY_TRAVERSALS.with(std::cell::Cell::get);
        for _ in 0..64 {
            with_runtime_builtin_bindings_by_name("SIN", |entries| {
                assert_eq!(entries.as_ptr(), first);
                assert_eq!(entries.len(), 1);
            });
        }
        assert_eq!(INVENTORY_TRAVERSALS.with(std::cell::Cell::get), before);
    }

    #[test]
    fn binding_range_retains_duplicates_and_retry_order() {
        let binding = runtime_builtin_bindings_by_name("sin")[0];
        let mut upper = binding;
        upper.identity.builtin.name = "SIN";
        let mut other = binding;
        other.identity.builtin.name = "z_other";
        let mut entries = vec![other, upper, binding, binding];
        entries.sort_by(|left, right| {
            compare_names(left.identity.builtin.name, right.identity.builtin.name)
        });
        let found = matching_bindings(&entries, "sIn");
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].identity.builtin.name, "SIN");
        assert_eq!(found[1].identity.builtin.name, "sin");
        assert!(matching_bindings(&entries, "missing").is_empty());
    }

    #[test]
    fn late_binding_registration_refreshes_early_miss_and_duplicates() {
        let binding = runtime_builtin_bindings_by_name("sin")[0];
        let mut entries = Vec::new();
        assert!(matching_bindings(&entries, "SIN").is_empty());
        insert_binding(&mut entries, binding);
        assert_eq!(matching_bindings(&entries, "SIN").len(), 1);
        insert_binding(&mut entries, binding);
        assert_eq!(matching_bindings(&entries, "SIN").len(), 2);
    }

    #[test]
    fn indexed_binding_lookup_matches_inventory_case_alias_and_order() {
        let inventory = runtime_builtin_bindings();
        let names = inventory
            .iter()
            .map(|entry| entry.identity.builtin.name)
            .chain(
                runmat_builtins::builtin_catalog_aliases()
                    .iter()
                    .map(|entry| entry.alias.name),
            );
        for name in names {
            for spelling in [name.to_owned(), name.to_ascii_uppercase()] {
                let expected: Vec<_> = inventory
                    .iter()
                    .filter(|entry| entry.identity.builtin.name.eq_ignore_ascii_case(&spelling))
                    .collect();
                with_runtime_builtin_bindings_by_name(&spelling, |actual| {
                    assert_eq!(actual.len(), expected.len(), "{spelling}");
                    for (actual, expected) in actual.iter().zip(&expected) {
                        assert_eq!(actual.identity, expected.identity);
                        assert!(std::ptr::fn_addr_eq(
                            actual.implementation,
                            expected.implementation
                        ));
                    }
                });
            }
        }
    }
}
