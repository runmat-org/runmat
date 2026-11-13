//! Registration authority for the exact process-global binding selected by a call.
//! This is a same-session generation, not an installed-code or persisted-profile ID.

use super::RuntimeBuiltinBinding;
use runmat_execution::Digest;

/// A binding and its registration witness are returned together. A caller
/// cannot attach another binding to this witness.
#[derive(Clone, Copy, Debug)]
pub struct RegisteredRuntimeBuiltinBinding {
    binding: RuntimeBuiltinBinding,
    registration: RuntimeBindingRegistration,
}

impl RegisteredRuntimeBuiltinBinding {
    pub fn binding(self) -> RuntimeBuiltinBinding {
        self.binding
    }

    pub fn registration(self) -> RuntimeBindingRegistration {
        self.registration
    }
}

/// Opaque, process-local witness of one selected inventory row.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeBindingRegistration {
    binding: RuntimeBuiltinBinding,
    index: usize,
    generation: u64,
}

impl RuntimeBindingRegistration {
    /// Validate that a caller did not detach this witness from the exact
    /// selected binding. Addresses are compared, never hashed into revision.
    pub fn certifies(self, selected: RuntimeBuiltinBinding) -> bool {
        self.binding.identity == selected.identity
            && self.binding.provenance == selected.provenance
            && std::ptr::fn_addr_eq(self.binding.implementation, selected.implementation)
            && match (
                self.binding.continuation_layout,
                selected.continuation_layout,
            ) {
                (Some(installed), Some(candidate)) => std::ptr::fn_addr_eq(installed, candidate),
                (None, None) => true,
                _ => false,
            }
    }

    /// Input to a session-owned RuntimeBindingSession revision. This is not
    /// suitable for cross-process persistence or installed-code attestation.
    pub fn session_generation(self) -> Digest {
        Digest::sha256_serialized(&(
            "runmat.runtime-binding-registration.v1",
            self.index,
            self.generation,
        ))
        .expect("typed registration witness serializes")
    }

    /// Check again at completion before publishing feedback. Function-address
    /// equality only verifies that the selected row is still the same row;
    /// no address is included in the authority digest.
    pub fn is_current(self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.is_current_in(&super::NAME_INDEX, 0)
        }
        #[cfg(target_arch = "wasm32")]
        {
            super::wasm_registry::with_index_and_generation(|entries, generation| {
                self.is_current_in(entries, generation)
            })
        }
    }

    fn is_current_in(self, entries: &[RuntimeBuiltinBinding], generation: u64) -> bool {
        generation == self.generation
            && entries
                .get(self.index)
                .is_some_and(|current| self.certifies(*current))
    }
}

/// Borrow one live inventory snapshot, retaining duplicate-name priority and
/// a witness for each exact row. A later browser registration revokes all
/// earlier witnesses by changing the generation.
pub fn registered_runtime_builtin_bindings_by_name(
    name: &str,
) -> Vec<RegisteredRuntimeBuiltinBinding> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        matching(&super::NAME_INDEX, 0, name)
    }
    #[cfg(target_arch = "wasm32")]
    {
        super::wasm_registry::with_index_and_generation(|entries, generation| {
            matching(entries, generation, name)
        })
    }
}

fn matching(
    entries: &[RuntimeBuiltinBinding],
    generation: u64,
    name: &str,
) -> Vec<RegisteredRuntimeBuiltinBinding> {
    super::matching_binding_range(entries, name)
        .map(|index| {
            let binding = entries[index];
            RegisteredRuntimeBuiltinBinding {
                binding,
                registration: RuntimeBindingRegistration {
                    binding,
                    index,
                    generation,
                },
            }
        })
        .collect()
}

/// The browser's mutable index and its checked registration generation.
/// The same owner is exercised without mutating process-global test state.
#[cfg(any(test, target_arch = "wasm32"))]
#[derive(Default)]
pub(super) struct RegistrationIndex {
    pub(super) entries: Vec<RuntimeBuiltinBinding>,
    pub(super) generation: u64,
}

#[cfg(any(test, target_arch = "wasm32"))]
impl RegistrationIndex {
    pub(super) fn insert(&mut self, binding: RuntimeBuiltinBinding) {
        assert!(
            self.try_insert(binding),
            "runtime binding registration generation exhausted"
        );
    }

    fn try_insert(&mut self, binding: RuntimeBuiltinBinding) -> bool {
        let Some(next) = self.generation.checked_add(1) else {
            return false;
        };
        super::insert_binding(&mut self.entries, binding);
        self.generation = next;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected() -> RuntimeBuiltinBinding {
        crate::builtins::wasm_registry::register_all();
        super::super::runtime_builtin_bindings_by_name("strcmp")[0]
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn selected_live_registration_witness_matches_exact_binding() {
        crate::builtins::wasm_registry::register_all();
        let row = registered_runtime_builtin_bindings_by_name("strcmp")
            .into_iter()
            .next()
            .expect("compiled strcmp binding");
        assert_eq!(row.binding().identity, selected().identity);
        assert!(std::ptr::fn_addr_eq(
            row.binding().implementation,
            selected().implementation
        ));
        assert!(row.registration().is_current());
        assert!(row.registration().certifies(row.binding()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn checked_mutation_revokes_pending_witness_even_for_same_binding() {
        let binding = selected();
        let mut index = RegistrationIndex::default();
        index.insert(binding);
        let witness = matching(&index.entries, index.generation, "strcmp")[0].registration();
        assert!(witness.is_current_in(&index.entries, index.generation));
        let before = witness.session_generation();
        index.insert(binding);
        assert!(!witness.is_current_in(&index.entries, index.generation));
        let after = matching(&index.entries, index.generation, "strcmp")[0]
            .registration()
            .session_generation();
        assert_ne!(before, after);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn duplicate_rows_have_distinct_selected_registration_witnesses() {
        let binding = selected();
        let mut index = RegistrationIndex::default();
        index.insert(binding);
        index.insert(binding);
        let rows = matching(&index.entries, index.generation, "strcmp");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row
            .registration()
            .is_current_in(&index.entries, index.generation)));
        assert_ne!(
            rows[0].registration().session_generation(),
            rows[1].registration().session_generation()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn witness_rejects_foreign_identity_and_implementation() {
        fn other_layout() -> Result<u64, crate::context::host_allocation::HostAllocationError> {
            Ok(1)
        }
        let binding = selected();
        let row = registered_runtime_builtin_bindings_by_name("strcmp")
            .into_iter()
            .next()
            .expect("compiled strcmp binding");
        let mut other_identity = binding;
        other_identity.identity.variant = "foreign";
        assert!(!row.registration().certifies(other_identity));
        let other_implementation = super::super::runtime_builtin_bindings_by_name("strcmpi")[0];
        let mut foreign_code = binding;
        foreign_code.implementation = other_implementation.implementation;
        assert!(!row.registration().certifies(foreign_code));
        let mut foreign_layout = binding;
        foreign_layout.continuation_layout = Some(other_layout);
        assert!(!row.registration().certifies(foreign_layout));
        let mut foreign_provenance = binding;
        foreign_provenance.provenance = None;
        assert!(!row.registration().certifies(foreign_provenance));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn exhausted_generation_does_not_mutate_registration_index() {
        let binding = selected();
        let mut index = RegistrationIndex {
            entries: vec![binding],
            generation: u64::MAX,
        };
        assert!(!index.try_insert(binding));
        assert_eq!(index.generation, u64::MAX);
        assert_eq!(index.entries.len(), 1);
        assert!(std::ptr::fn_addr_eq(
            index.entries[0].implementation,
            binding.implementation
        ));
    }
}
