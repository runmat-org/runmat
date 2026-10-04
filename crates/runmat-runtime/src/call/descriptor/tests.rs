use super::*;
use crate::context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts};
use std::{cell::Cell, rc::Rc};

thread_local! {
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static DIAGNOSTIC: Cell<&'static str> = const { Cell::new("RunMat:UndefinedFunction") };
}

struct FailingBuiltin;

impl RuntimeBuiltinService for FailingBuiltin {
    fn bindings_by_name(&self, name: &str) -> Vec<crate::builtin::RuntimeBuiltinBinding> {
        let Some(entry) = runmat_builtins::builtin_catalog_primary_entry_by_name(name) else {
            return Vec::new();
        };
        vec![crate::builtin::RuntimeBuiltinBinding::new(
            runmat_builtins::BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: "default",
            },
            |_| {
                CALLS.with(|calls| calls.set(calls.get() + 1));
                let diagnostic = DIAGNOSTIC.with(Cell::get);
                Box::pin(async move {
                    Err(crate::build_runtime_error("the selected callee failed")
                        .with_identifier(diagnostic)
                        .build())
                })
            },
        )]
    }
}

#[test]
fn selected_named_fallback_error_is_not_absence_or_replayed() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(FailingBuiltin)));
    for identifier in ["RunMat:UndefinedFunction", "RunMat:ProbeReceiverInvalid"] {
        CALLS.with(|calls| calls.set(0));
        DIAGNOSTIC.with(|diagnostic| diagnostic.set(identifier));
        let descriptor = CallableDescriptor::resolved(
            CallableIdentity::DynamicName(runmat_types::SymbolName("logical".into())),
            vec![Value::Num(1.0)],
            1,
            CallableFallbackPolicy::RuntimeNameResolution,
            CallableCallKind::Direct,
        );
        let err = futures::executor::block_on(try_execute_callable_descriptor_in_context(
            &runtime, descriptor,
        ))
        .expect_err("the selected builtin error must propagate");
        assert_eq!(err.identifier(), Some(identifier));
        assert_eq!(
            err.failure_origin(),
            crate::runtime_error::RuntimeFailureOrigin::Unclassified
        );
        CALLS.with(|calls| assert_eq!(calls.get(), 1));
    }
}

#[test]
fn absent_named_callable_is_a_producer_classified_language_failure() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()));
    let _scope = runtime.enter();
    let error = futures::executor::block_on(forward_named_fallback(
        Some(&runtime),
        "__runmat_missing_input_callable__".into(),
        Vec::new(),
        1,
    ))
    .expect_err("missing callable");
    assert_eq!(
        error.failure_origin(),
        crate::runtime_error::RuntimeFailureOrigin::LanguageEvaluation
    );
    assert_eq!(error.identifier(), Some("RunMat:UndefinedFunction"));
}

#[test]
fn unselected_named_fallback_is_absent_without_invocation() {
    let runtime = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(FailingBuiltin)));
    CALLS.with(|calls| calls.set(0));
    let descriptor = CallableDescriptor::resolved(
        CallableIdentity::DynamicName(runmat_types::SymbolName(
            "AbsentMethodResolutionFixture".into(),
        )),
        vec![Value::Num(1.0)],
        1,
        CallableFallbackPolicy::RuntimeNameResolution,
        CallableCallKind::Direct,
    );
    let result = futures::executor::block_on(try_execute_callable_descriptor_in_context(
        &runtime, descriptor,
    ))
    .expect("a missing candidate is not a callee error");
    assert!(result.is_none());
    CALLS.with(|calls| assert_eq!(calls.get(), 0));
}
