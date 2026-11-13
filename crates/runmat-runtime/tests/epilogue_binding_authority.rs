//! Inventory and implementation identity must be tested against one runtime library.
use runmat_runtime::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture},
    builtins::common::spec::{fusion_epilogue_operation, FusionEpilogueOperation},
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use std::rc::Rc;

struct Selected(Vec<RuntimeBuiltinBinding>);
impl RuntimeBuiltinService for Selected {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.0
            .iter()
            .filter(|binding| binding.identity.builtin.name == name)
            .copied()
            .collect()
    }
}
fn replacement(_: runmat_runtime::builtin::RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        let sequence = runmat_value::ValueSequence::single(runmat_value::Value::Num(99.0))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}
fn context(bindings: Vec<RuntimeBuiltinBinding>) -> RuntimeContext {
    RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(Selected(bindings))))
}

#[test]
fn exact_binding_allows_compiled_implementation_but_never_scoped_override() {
    runmat_gc::gc_test_context(|| {
        let name = runmat_builtins::POWER_CATALOG_ENTRY.identity.name;
        let original = runmat_runtime::builtin::runtime_builtin_bindings_by_name(name);
        assert_eq!(original.len(), 1);
        assert_eq!(
            fusion_epilogue_operation(name, Some(&context(original.clone()))),
            Some(FusionEpilogueOperation::RealPower)
        );
        let mut changed = original;
        changed[0].implementation = replacement;
        assert_eq!(
            fusion_epilogue_operation(name, Some(&context(changed))),
            None
        );
        assert_eq!(
            fusion_epilogue_operation(name, Some(&context(Vec::new()))),
            None
        );
    });
}

#[test]
fn scoped_authority_never_falls_back_to_joined_legacy_epilogue() {
    runmat_gc::gc_test_context(|| {
        assert_eq!(
            fusion_epilogue_operation("diag", None),
            Some(FusionEpilogueOperation::Diagonal)
        );
        assert_eq!(
            fusion_epilogue_operation("diag", Some(&context(Vec::new()))),
            None
        );
    });
}
