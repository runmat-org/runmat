use super::*;
use futures::FutureExt;
use runmat_native_codegen::{NativeBlockId, NativeFunction, NativeFunctionAbi};
use runmat_types::{ProgramFunctionId, ProgramSourceId};

fn workspace(values: BTreeMap<String, Value>) -> NativeWorkspaceState {
    workspace_for_context(values, None)
}

fn workspace_for_context(
    values: BTreeMap<String, Value>,
    context: Option<&runmat_runtime::context::RuntimeContext>,
) -> NativeWorkspaceState {
    let function = NativeFunction {
        spawn_safety: runmat_types::SpawnSafetyFact::SpawnSafe,
        lexical_bindings: Vec::new(),
        inherited_storage_bindings: Vec::new(),
        creates_lexical_environment: false,
        id: runmat_types::FunctionId(0),
        portable_id: ProgramFunctionId(0),
        mir_body_digest: runmat_execution::Digest::sha256(b"fixture:native-gc-workspace"),
        source: ProgramSourceId(0),
        name: "gc_workspace".into(),
        source_role: runmat_types::FunctionSourceRole::Unknown,
        lexical_parent: None,
        class_method_owner: None,
        capabilities: Default::default(),
        abi: NativeFunctionAbi {
            fixed_inputs: vec![],
            varargin: None,
            fixed_outputs: vec![],
            varargout: None,
            implicit_nargin: None,
            implicit_nargout: None,
        },
        argument_validations: vec![],
        binding_locals: vec![],
        closure_captures: vec![],
        locals: vec![],
        entry: NativeBlockId(0),
        blocks: vec![],
        expression_continuation_geometry:
            runmat_native_codegen::NativeExpressionContinuationGeometry {
                region_capacity: 0,
                retained_input_capacity: 0,
            },
        operation_contracts: vec![],
        expected_sites: vec![],
    };
    NativeWorkspaceState::new_with_runtime(
        &function,
        NativeWorkspaceInput::from_test_values(values, Default::default(), Vec::new(), Vec::new()),
        FrameGlobalBindings::default(),
        None,
        None,
        context,
    )
    .unwrap()
}

fn handle_value(handle: runmat_gc::GcHandle) -> Value {
    Value::HandleObject(runmat_value::HandleRef {
        class_name: "WorkspaceGcFixture".into(),
        target: handle,
        valid: true,
    })
}

#[test]
fn assigned_unmapped_value_survives_suspension_and_state_drop_releases_roots() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::assigned_unmapped_value_survives_suspension_and_state_drop_releases_roots"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let state = workspace(BTreeMap::new());
        let service = state.service();
        let allocation =
            runmat_gc::gc_allocate_rooted(Value::String("pending workspace".into())).unwrap();
        let handle = allocation.handle();
        let assigned = handle_value(handle);
        let expected = assigned.clone();
        let pending_service = Rc::clone(&service);
        let mut operation = Box::pin(async move {
            pending_service.assign("unmapped", assigned).unwrap();
            allocation.unroot().unwrap();
            let mut yielded = false;
            futures::future::poll_fn(|cx| {
                if yielded {
                    std::task::Poll::Ready(())
                } else {
                    yielded = true;
                    cx.waker().wake_by_ref();
                    std::task::Poll::Pending
                }
            })
            .await;
            pending_service.lookup("unmapped").unwrap()
        });
        assert!(operation.as_mut().now_or_never().is_none());
        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).unwrap(),
            Value::String("pending workspace".into())
        );
        assert_eq!(futures::executor::block_on(operation), expected);
        drop(state);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_err(),
            "executor owner drop releases frame roots even while a stale service remains"
        );
        let error = service.try_lookup("unmapped").unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Workspace:FrameClosed"));
        drop(service);
    });
}

#[test]
fn stale_service_does_not_retain_the_canonical_projection_after_state_drop() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::stale_service_does_not_retain_the_canonical_projection_after_state_drop"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let allocation = runmat_gc::gc_allocate_rooted(Value::String("canonical".into())).unwrap();
        let handle = allocation.handle();
        let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
        let mut initial = runmat_runtime::workspace::WorkspaceDelta::new();
        initial
            .assign("canonical_only", handle_value(handle))
            .unwrap();
        owner.apply(initial).unwrap();
        let projection = Rc::new(
            runmat_runtime::workspace::BaseWorkspaceProjectionService::new(
                owner.begin_projection().unwrap(),
            ),
        );
        let canonical: Rc<dyn RuntimeWorkspaceService> = projection.clone();
        let function = NativeFunction {
            spawn_safety: runmat_types::SpawnSafetyFact::SpawnSafe,
            lexical_bindings: Vec::new(),
            inherited_storage_bindings: Vec::new(),
            creates_lexical_environment: false,
            id: runmat_types::FunctionId(0),
            portable_id: ProgramFunctionId(0),
            mir_body_digest: runmat_execution::Digest::sha256(
                b"fixture:native-gc-workspace-canonical",
            ),
            source: ProgramSourceId(0),
            name: "gc_workspace".into(),
            source_role: runmat_types::FunctionSourceRole::Unknown,
            lexical_parent: None,
            class_method_owner: None,
            capabilities: Default::default(),
            abi: NativeFunctionAbi {
                fixed_inputs: vec![],
                varargin: None,
                fixed_outputs: vec![],
                varargout: None,
                implicit_nargin: None,
                implicit_nargout: None,
            },
            argument_validations: vec![],
            binding_locals: vec![],
            closure_captures: vec![],
            locals: vec![],
            entry: NativeBlockId(0),
            blocks: vec![],
            expression_continuation_geometry:
                runmat_native_codegen::NativeExpressionContinuationGeometry {
                    region_capacity: 0,
                    retained_input_capacity: 0,
                },
            operation_contracts: vec![],
            expected_sites: vec![],
        };
        let state = NativeWorkspaceState::new(
            &function,
            NativeWorkspaceInput::from_test_values(
                Default::default(),
                Default::default(),
                Vec::new(),
                Vec::new(),
            ),
            FrameGlobalBindings::default(),
            Some(canonical.clone()),
            None,
        )
        .unwrap();
        let service = state.service();
        drop(canonical);
        drop(projection);
        drop(owner);
        allocation.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());

        drop(state);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_err(),
            "a stale native service must not retain its canonical projection"
        );
        assert_eq!(
            service.try_snapshot().unwrap_err().identifier(),
            Some("RunMat:Workspace:FrameClosed")
        );
    });
}

#[test]
fn input_snapshot_aliases_remove_clear_and_replacement_have_exact_root_lifetimes() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::input_snapshot_aliases_remove_clear_and_replacement_have_exact_root_lifetimes"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let allocation = runmat_gc::gc_allocate_rooted(Value::String("aliased".into())).unwrap();
        let handle = allocation.handle();
        let nested =
            Value::Cell(runmat_value::CellArray::new(vec![handle_value(handle)], 1, 1).unwrap());
        let state = workspace(BTreeMap::from([
            ("a".into(), nested.clone()),
            ("b".into(), nested),
        ]));
        let service = state.service();
        allocation.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        service.remove("a").unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_ok(),
            "independent alias root survives removal"
        );
        service.assign("b", Value::Num(0.0)).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_err(),
            "replacement releases last root"
        );

        let allocation = runmat_gc::gc_allocate_rooted(Value::Num(17.0)).unwrap();
        let handle = allocation.handle();
        service.assign("clear_me", handle_value(handle)).unwrap();
        allocation.unroot().unwrap();
        service.clear().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());

        // A captured generation retains its own old roots. Clearing the live
        // workspace releases them only after that external reader retires.
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let state = workspace_for_context(BTreeMap::new(), Some(&runtime));
        let service = state.service();
        let allocation = runmat_gc::gc_allocate_rooted(Value::Num(29.0)).unwrap();
        let handle = allocation.handle();
        service
            .assign_with_context(&runtime, "captured", handle_value(handle))
            .unwrap();
        let capture = service
            .capture_workspace_generation_with_context(&runtime)
            .unwrap()
            .expect("private workspace generation capture");
        allocation.unroot().unwrap();
        service.clear_with_context(&runtime).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        assert_eq!(capture.get("captured"), Some(&handle_value(handle)));
        drop(capture);
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_err(),
            "only the external capture retains the cleared generation"
        );
    });
}

#[test]
fn reentrant_mutation_is_structured_and_gc_never_borrows_workspace_storage() {
    if runmat_gc::test_support::run_isolated_gc_test(concat!(
        module_path!(),
        "::reentrant_mutation_is_structured_and_gc_never_borrows_workspace_storage"
    )) {
        return;
    }
    runmat_gc::gc_test_context(|| {
        let allocation = runmat_gc::gc_allocate_rooted(Value::Num(19.0)).unwrap();
        let handle = allocation.handle();
        let state = workspace(BTreeMap::from([("retained".into(), handle_value(handle))]));
        let service = state.service();
        allocation.unroot().unwrap();
        let borrow = state.frame.lexical.borrow_mut();
        runmat_gc::gc_collect_major().unwrap();
        assert!(
            runmat_gc::gc_clone_value(&handle).is_ok(),
            "retained generation roots do not scan borrowed Native metadata"
        );
        for result in [
            service.assign("retained", Value::Num(0.0)),
            service.remove("retained"),
            service.clear(),
        ] {
            assert_eq!(
                result.unwrap_err().identifier(),
                Some("RunMat:NativeWorkspaceBorrow")
            );
        }
        assert!(
            !state.needs_reconciliation(),
            "failed mutation leaves both storage and dirty status intact"
        );
        drop(borrow);
        assert_eq!(service.lookup("retained"), Some(handle_value(handle)));
        service.clear().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
