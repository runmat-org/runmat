//! Selection is owned by the actual Runtime binding, including its physical
//! declaration association. Shared process-wide provenance is evidence, not
//! permission to lower a different scoped implementation.
use runmat_runtime::{
    builtin::{RuntimeBuiltinBinding, RuntimeBuiltinFuture},
    builtins::common::spec::{
        canonical_fusion_declaration, selected_builtin_binding, selected_fusion_binding,
    },
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
};
use runmat_value::Value;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

struct ChangingService {
    bindings: RefCell<Vec<RuntimeBuiltinBinding>>,
    calls: Cell<usize>,
}

impl RuntimeBuiltinService for ChangingService {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding> {
        self.calls.set(self.calls.get() + 1);
        self.bindings
            .borrow()
            .iter()
            .copied()
            .filter(|binding| binding.identity.builtin.name.eq_ignore_ascii_case(name))
            .collect()
    }
}

fn replacement(_: runmat_runtime::builtin::RuntimeBuiltinInvocation<'_>) -> RuntimeBuiltinFuture {
    runmat_runtime::builtin::admitted_builtin_future(None, "fixture", async {
        let sequence = runmat_value::ValueSequence::single(Value::Num(99.0))
            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
        runmat_runtime::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

#[test]
fn indexed_declarations_never_cache_scoped_selection_permission() {
    runmat_gc::gc_test_context(|| {
        let original = runmat_runtime::builtin::runtime_builtin_bindings_by_name("sin");
        assert_eq!(original.len(), 1, "compiled sin bindings: {original:?}");
        let entry = runmat_builtins::builtin_catalog_entry_by_name("sin").unwrap();
        assert_eq!(entry.bindings.len(), 1, "catalog sin binding cardinality");
        assert_eq!(
            original[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
        let provenance =
            runmat_builtins::with_builtin_implementation_provenance_by_name("sin", |entries| {
                entries
                    .iter()
                    .copied()
                    .filter(|entry| {
                        entry.authority
                            == runmat_builtins::BuiltinImplementationAuthority::CanonicalBinding
                            && entry.binding_variant == Some("default")
                    })
                    .collect::<Vec<_>>()
            });
        assert_eq!(
            provenance.len(),
            1,
            "single-library sin provenance: {provenance:#?}"
        );
        let declaration = canonical_fusion_declaration(original[0].identity).unwrap_or_else(|| {
            panic!("sin declaration did not join unique provenance: {provenance:#?}")
        });
        assert_eq!(declaration.binding, original[0].identity);
        assert!(declaration.spec.elementwise.is_some());
        runmat_runtime::builtin::with_runtime_builtin_bindings_by_name("sin", |compiled| {
            assert_eq!(compiled.len(), 1);
            assert!(std::ptr::fn_addr_eq(
                compiled[0].implementation,
                original[0].implementation
            ));
        });
        let service = Rc::new(ChangingService {
            bindings: RefCell::new(original.clone()),
            calls: Cell::new(0),
        });
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(service.clone()));
        let expected = Some(original[0].identity);
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_some());
        service.bindings.borrow_mut()[0].provenance = None;
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_none());
        *service.bindings.borrow_mut() = original.clone();
        service.bindings.borrow_mut()[0].implementation = replacement;
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_none());
        *service.bindings.borrow_mut() = vec![original[0], original[0]];
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_none());
        service.bindings.borrow_mut().clear();
        assert!(selected_fusion_binding("sin", expected, Some(&context)).is_none());
        *service.bindings.borrow_mut() = original;
        assert!(selected_fusion_binding("SIN", expected, Some(&context)).is_some());
        assert_eq!(service.calls.get(), 6);
        // A separate invocation authority cannot inherit this service's result.
        let absent = Rc::new(ChangingService {
            bindings: RefCell::new(vec![]),
            calls: Cell::new(0),
        });
        let other = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(absent.clone()));
        assert!(selected_fusion_binding("sin", expected, Some(&other)).is_none());
        assert_eq!(absent.calls.get(), 1);
        assert_eq!(service.calls.get(), 6);
    });
}

#[test]
fn canonical_selection_is_independent_of_fusion_and_rechecks_scoped_implementation() {
    runmat_gc::gc_test_context(|| {
        let original = runmat_runtime::builtin::runtime_builtin_bindings_by_name("log");
        assert_eq!(original.len(), 1, "compiled log bindings: {original:?}");
        let entry = runmat_builtins::builtin_catalog_entry_by_name("log").unwrap();
        assert_eq!(entry.bindings.len(), 1);
        assert_eq!(
            original[0].identity,
            entry.binding_identity(&entry.bindings[0])
        );
        let service = Rc::new(ChangingService {
            bindings: RefCell::new(original.clone()),
            calls: Cell::new(0),
        });
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(service.clone()));
        let expected = Some(original[0].identity);
        let selected = selected_builtin_binding("log", expected, Some(&context)).unwrap();
        assert!(std::ptr::fn_addr_eq(
            selected.implementation,
            original[0].implementation
        ));
        service.bindings.borrow_mut()[0].implementation = replacement;
        assert!(selected_builtin_binding("log", expected, Some(&context)).is_none());
        *service.bindings.borrow_mut() = vec![original[0], original[0]];
        assert!(selected_builtin_binding("log", expected, Some(&context)).is_none());
        service.bindings.borrow_mut().clear();
        assert!(selected_builtin_binding("log", expected, Some(&context)).is_none());
        *service.bindings.borrow_mut() = original;
        assert!(selected_builtin_binding("LOG", expected, Some(&context)).is_some());
        assert_eq!(service.calls.get(), 5);
    });
}
