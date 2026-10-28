use std::sync::atomic::{AtomicU64, Ordering};
use std::{collections::BTreeMap, rc::Rc};

use runmat_execution::Digest;
use runmat_native_codegen::aot::{AotProgramManifest, AotRuntimeBindingMode};
use runmat_runtime::{builtin::RuntimeBuiltinBinding, context::RuntimeBuiltinService};

use crate::input::AotBuiltinResolver;

pub fn resolve(
    program: &AotProgramManifest,
    resolver: AotBuiltinResolver,
) -> Result<Option<Rc<dyn RuntimeBuiltinService>>, String> {
    if program.runtime_binding_mode == AotRuntimeBindingMode::Dynamic {
        return Ok(None);
    }
    let mut bindings = Vec::with_capacity(program.builtin_bindings.len());
    for (index, expected) in program.builtin_bindings.iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| "standalone builtin binding index exceeds its ABI".to_string())?;
        // SAFETY: the generated resolver returns the address of an immutable
        // RuntimeBuiltinBinding static from this exact runtime archive.
        let address = unsafe { resolver(index) };
        if address.is_null() {
            return Err(format!(
                "standalone object cannot resolve builtin {}#{}",
                expected.name, expected.variant
            ));
        }
        // SAFETY: the resolver ABI admits only RuntimeBuiltinBinding statics;
        // the value is copied immediately and its identity is checked below.
        let binding = unsafe { *(address.cast::<RuntimeBuiltinBinding>()) };
        if binding.identity.builtin.name != expected.name
            || binding.identity.variant != expected.variant
        {
            return Err(format!(
                "standalone builtin binding {}#{} does not match its linked symbol",
                expected.name, expected.variant
            ));
        }
        bindings.push(binding);
    }
    Ok(Some(Rc::new(ImmutableBuiltinBindings::new(bindings)?)))
}

/// Capture the stock archive's immutable native inventory for an explicitly
/// admitted hosted-dispatch invocation. This table owns its registration
/// identity; lookups never fall back to another process-global table.
#[cfg(feature = "stock-hosted-factories")]
pub(super) fn hosted_bindings() -> Result<Rc<dyn RuntimeBuiltinService>, String> {
    Ok(Rc::new(ImmutableBuiltinBindings::new(
        runmat_runtime::builtin::runtime_builtin_bindings(),
    )?))
}

struct ImmutableBuiltinBindings {
    by_name: BTreeMap<String, Vec<RuntimeBuiltinBinding>>,
    /// Nonpersistable identity for this immutable resolved binding table.
    instance: u64,
}

static NEXT_BINDING_TABLE_INSTANCE: AtomicU64 = AtomicU64::new(1);

impl ImmutableBuiltinBindings {
    fn new(bindings: Vec<RuntimeBuiltinBinding>) -> Result<Self, String> {
        let mut by_name = BTreeMap::<String, Vec<RuntimeBuiltinBinding>>::new();
        for binding in bindings {
            let variants = by_name
                .entry(binding.identity.builtin.name.to_ascii_lowercase())
                .or_default();
            if variants
                .iter()
                .any(|existing| existing.identity == binding.identity)
            {
                return Err(format!(
                    "standalone builtin binding {}#{} is duplicated",
                    binding.identity.builtin.name, binding.identity.variant
                ));
            }
            variants.push(binding);
        }
        let instance = NEXT_BINDING_TABLE_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("immutable builtin registration identity exhausted");
        Ok(Self { by_name, instance })
    }
}

impl RuntimeBuiltinService for ImmutableBuiltinBindings {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.by_name
            .get(&name.to_ascii_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    fn binding_registration_revision(&self, binding: RuntimeBuiltinBinding) -> Option<Digest> {
        let members = self
            .by_name
            .get(&binding.identity.builtin.name.to_ascii_lowercase())?;
        members
            .iter()
            .any(|member| {
                member.identity == binding.identity
                    && member.provenance == binding.provenance
                    && std::ptr::fn_addr_eq(member.implementation, binding.implementation)
                    && match (member.continuation_layout, binding.continuation_layout) {
                        (Some(installed), Some(candidate)) => {
                            std::ptr::fn_addr_eq(installed, candidate)
                        }
                        (None, None) => true,
                        _ => false,
                    }
            })
            .then(|| {
                Digest::sha256_serialized(&(
                    "runmat.aot-immutable-binding-registration.v1",
                    self.instance,
                ))
                .expect("typed immutable registration scope serializes")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::ImmutableBuiltinBindings;
    use runmat_runtime::context::RuntimeBuiltinService;

    #[test]
    fn closed_world_service_never_falls_back_for_an_absent_name() {
        let service = ImmutableBuiltinBindings::new(Vec::new()).expect("empty exact registry");
        assert!(service.bindings_by_name("abs").is_empty());
    }

    #[test]
    fn immutable_instance_revision_certifies_only_its_exact_selected_binding() {
        let binding = runmat_runtime::builtin::runtime_builtin_bindings_by_name("strcmp")[0];
        let other = runmat_runtime::builtin::runtime_builtin_bindings_by_name("strcmpi")[0];
        let first = ImmutableBuiltinBindings::new(vec![binding]).expect("exact binding table");
        let second = ImmutableBuiltinBindings::new(vec![binding]).expect("distinct exact table");
        assert!(first.binding_registration_revision(binding).is_some());
        assert_ne!(
            first.binding_registration_revision(binding),
            second.binding_registration_revision(binding)
        );
        assert_eq!(first.binding_registration_revision(other), None);
        let mut foreign_code = binding;
        foreign_code.implementation = other.implementation;
        assert_eq!(first.binding_registration_revision(foreign_code), None);
        let mut missing_layout = binding;
        assert!(missing_layout.continuation_layout.take().is_some());
        assert_eq!(first.binding_registration_revision(missing_layout), None);
    }
}
