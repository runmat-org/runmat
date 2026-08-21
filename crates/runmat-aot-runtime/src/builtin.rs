use std::{collections::BTreeMap, rc::Rc};

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
    Ok(Some(Rc::new(ClosedWorldBuiltins::new(bindings)?)))
}

struct ClosedWorldBuiltins {
    by_name: BTreeMap<String, Vec<RuntimeBuiltinBinding>>,
}

impl ClosedWorldBuiltins {
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
        Ok(Self { by_name })
    }
}

impl RuntimeBuiltinService for ClosedWorldBuiltins {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.by_name
            .get(&name.to_ascii_lowercase())
            .cloned()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::ClosedWorldBuiltins;
    use runmat_runtime::context::RuntimeBuiltinService;

    #[test]
    fn closed_world_service_never_falls_back_for_an_absent_name() {
        let service = ClosedWorldBuiltins::new(Vec::new()).expect("empty exact registry");
        assert!(service.bindings_by_name("abs").is_empty());
    }
}
