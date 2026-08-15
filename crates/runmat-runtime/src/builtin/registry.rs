use super::RuntimeBuiltinBinding;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(not(target_arch = "wasm32"))]
inventory::collect!(RuntimeBuiltinBinding);

#[cfg(not(target_arch = "wasm32"))]
pub fn runtime_builtin_bindings() -> Vec<RuntimeBuiltinBinding> {
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
    }

    pub fn submit(binding: RuntimeBuiltinBinding) {
        BINDINGS.with_borrow_mut(|bindings| bindings.push(Box::leak(Box::new(binding))));
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
    runtime_builtin_bindings()
        .into_iter()
        .filter(|binding| binding.identity.builtin.name.eq_ignore_ascii_case(name))
        .collect()
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
            declared.insert(binding.identity);
            let count = actual_counts.get(&binding.identity).copied().unwrap_or(0);
            if matches!(
                binding.availability,
                runmat_builtins::BuiltinBindingAvailability::Required
            ) && count != 1
            {
                errors.push(registry_error(
                    &binding.identity,
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
}
