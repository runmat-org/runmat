use super::*;
use runmat_native_codegen::{NativeLexicalBinding, NativeLexicalBindingKind, NativeLocalId};
use std::error::Error;

#[test]
fn committed_native_workspace_remove_tombstones_its_owned_lexical_cell() {
    runmat_gc::gc_test_context(|| {
        let (mut function, input) = super::tests::fixture(Value::Num(7.0));
        function.source_role = runmat_types::FunctionSourceRole::Nested;
        function.creates_lexical_environment = true;
        function.lexical_bindings = vec![NativeLexicalBinding {
            binding: BindingId(0),
            local: NativeLocalId(0),
            kind: NativeLexicalBindingKind::Owned,
        }];
        let state =
            NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
        let (environment, _roots) =
            runmat_runtime::call::lexical::allocate_environment(None).unwrap();
        assert!(
            runmat_runtime::call::lexical::reserve_owned_binding(&environment, BindingId(0))
                .unwrap()
        );
        runmat_runtime::call::lexical::write_owned_binding(
            &environment,
            BindingId(0),
            Value::Num(7.0),
        )
        .unwrap();
        state
            .attach_lexical_bindings(&function, Some(&environment))
            .unwrap();
        state.service().assign("missing", Value::Num(8.0)).unwrap();
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, BindingId(0)).unwrap(),
            Some(Value::Num(8.0))
        );
        state.service().remove("missing").unwrap();
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, BindingId(0)).unwrap(),
            None
        );
        runmat_runtime::call::lexical::write_owned_binding(
            &environment,
            BindingId(0),
            Value::Num(11.0),
        )
        .unwrap();
        assert_eq!(
            state.service().try_lookup("missing").unwrap(),
            Some(Value::Num(11.0))
        );
    });
}

#[test]
fn native_anonymous_transit_clear_does_not_retire_parent_cell() {
    runmat_gc::gc_test_context(|| {
        let (mut function, input) = super::tests::fixture(Value::Num(3.0));
        function.source_role = runmat_types::FunctionSourceRole::Anonymous;
        function.lexical_bindings = vec![NativeLexicalBinding {
            binding: BindingId(0),
            local: NativeLocalId(0),
            kind: NativeLexicalBindingKind::Transit,
        }];
        let state =
            NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
        let (environment, _roots) =
            runmat_runtime::call::lexical::allocate_environment(None).unwrap();
        assert!(
            runmat_runtime::call::lexical::reserve_owned_binding(&environment, BindingId(0))
                .unwrap()
        );
        runmat_runtime::call::lexical::write_owned_binding(
            &environment,
            BindingId(0),
            Value::Num(9.0),
        )
        .unwrap();
        state
            .attach_lexical_bindings(&function, Some(&environment))
            .unwrap();
        state.service().assign("missing", Value::Num(4.0)).unwrap();
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, BindingId(0)).unwrap(),
            Some(Value::Num(9.0))
        );
        state.service().clear().unwrap();
        assert_eq!(
            runmat_runtime::call::lexical::read_binding(&environment, BindingId(0)).unwrap(),
            Some(Value::Num(9.0))
        );
    });
}

#[test]
fn committed_native_assignment_retains_lexical_failure_as_a_typed_source() {
    runmat_gc::gc_test_context(|| {
        let context = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        context.set_error_namespace("Acme");
        let _scope = context.enter();
        let (mut function, input) = super::tests::fixture(Value::Num(7.0));
        function.source_role = runmat_types::FunctionSourceRole::Nested;
        function.creates_lexical_environment = true;
        function.lexical_bindings = vec![NativeLexicalBinding {
            binding: BindingId(0),
            local: NativeLocalId(0),
            kind: NativeLexicalBindingKind::Owned,
        }];
        let state =
            NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
        let (environment, _roots) =
            runmat_runtime::call::lexical::allocate_environment(None).unwrap();
        // Deliberately omit the Owned reservation: the shared lexical cell is
        // this name's sole value authority, so failure cannot publish a value.
        state
            .attach_lexical_bindings(&function, Some(&environment))
            .unwrap();
        let error = state
            .service()
            .assign("missing", Value::Num(8.0))
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("Acme:Workspace:LexicalPublication")
        );
        assert_eq!(
            error.gpu_gather_retry,
            runmat_runtime::runtime_error::GpuGatherRetry::Never
        );
        let cause = error
            .source()
            .and_then(|source| source.downcast_ref::<runmat_runtime::RuntimeError>())
            .expect("original lexical RuntimeError is retained");
        assert_eq!(cause.identifier(), Some("Acme:LexicalEnvironment"));
        assert_eq!(state.service().try_lookup("missing").unwrap(), None);
    });
}
