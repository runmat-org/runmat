use super::{
    NativeWorkspaceBinding, NativeWorkspaceGlobalBinding, NativeWorkspaceInput,
    NativeWorkspaceState,
};
use runmat_native_codegen::{
    NativeBlockId, NativeFunction, NativeFunctionAbi, NativeLocalId, NativeLocalKind,
    NativeLocalMetadata,
};
use runmat_runtime::workspace::WorkspaceDetachSelection;
use runmat_types::{BindingId, ProgramFunctionId, ProgramSourceId};
use runmat_value::{Tensor, Value};
use std::collections::BTreeMap;

fn initial_value(state: &NativeWorkspaceState, slot: usize) -> Option<Value> {
    state.local_names.name(slot).and_then(|name| {
        state
            .service()
            .try_lookup(name)
            .expect("fixture workspace read")
    })
}

fn snapshot_values(state: &NativeWorkspaceState) -> BTreeMap<String, Value> {
    state
        .service()
        .try_snapshot()
        .expect("fixture workspace snapshot")
        .into_iter()
        .collect()
}

#[test]
fn retired_native_frame_preserves_closed_error_with_and_without_retained_storage() {
    for retain_storage in [false, true] {
        let (function, input) = fixture(Value::Num(1.0));
        let state =
            NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
        let service = state.service();
        let retained = retain_storage.then(|| state.frame.clone());
        drop(state);

        for error in [
            service.try_lookup("missing").unwrap_err(),
            service.try_snapshot().unwrap_err(),
            service.assign("missing", Value::Num(2.0)).unwrap_err(),
        ] {
            assert_eq!(error.identifier(), Some("RunMat:Workspace:FrameClosed"));
            assert_eq!(
                error.to_string(),
                super::native_workspace_closed().to_string()
            );
        }
        drop(retained);
    }
}

#[test]
fn only_external_workspace_mutation_requests_reconciliation() {
    let (function, input) = fixture(Value::Num(1.0));
    let state =
        NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
    assert!(!state.needs_reconciliation());
    state.synchronize_local(0, Value::Num(2.0)).unwrap();
    assert!(!state.needs_reconciliation());
    let service = state.service();
    service.assign("missing", Value::Num(3.0)).unwrap();
    assert!(state.needs_reconciliation());
    state.mark_reconciled();
    service.remove("missing").unwrap();
    assert!(state.needs_reconciliation());
    state.mark_reconciled();
    service.clear().unwrap();
    assert!(state.needs_reconciliation());
}

#[test]
fn exact_native_detach_keeps_old_value_and_does_not_erase_rebinding() {
    let (function, input) = fixture(Value::Num(1.0));
    let state =
        NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
    let service = state.service();
    let old = service
        .detach_selected(&WorkspaceDetachSelection::Names(vec!["missing".into()]))
        .expect("detach before callback");
    assert_eq!(old.entries(), &[("missing".into(), Value::Num(1.0))]);
    service.assign("missing", Value::Num(2.0)).unwrap();
    assert_eq!(
        service.try_lookup("missing").unwrap(),
        Some(Value::Num(2.0))
    );
    drop(old);
    assert_eq!(
        service.try_lookup("missing").unwrap(),
        Some(Value::Num(2.0))
    );
}

pub(super) fn fixture(value: Value) -> (NativeFunction, NativeWorkspaceInput) {
    let binding = BindingId(0);
    let function = NativeFunction {
        spawn_safety: runmat_types::SpawnSafetyFact::SpawnSafe,
        lexical_bindings: Vec::new(),
        inherited_storage_bindings: Vec::new(),
        creates_lexical_environment: false,
        id: runmat_types::FunctionId(0),
        portable_id: ProgramFunctionId(0),
        mir_body_digest: runmat_execution::Digest::sha256(b"fixture:native-workspace-snapshot"),
        source: ProgramSourceId(0),
        name: "snapshot".into(),
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
        binding_locals: vec![runmat_native_codegen::NativeBindingLocal {
            binding,
            local: NativeLocalId(0),
        }],
        closure_captures: vec![],
        locals: vec![NativeLocalMetadata {
            id: NativeLocalId(0),
            binding: Some(binding),
            name: Some("missing".into()),
            kind: NativeLocalKind::Binding,
        }],
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
    let input = NativeWorkspaceInput::from_test_values(
        BTreeMap::from([("missing".into(), value)]),
        BTreeMap::from([(binding, "missing".into())]),
        vec![NativeWorkspaceBinding {
            binding,
            name: "missing".into(),
        }],
        Vec::new(),
    );
    (function, input)
}

fn error(function: &NativeFunction, input: NativeWorkspaceInput) -> String {
    match NativeWorkspaceState::new(function, input, Default::default(), None, None) {
        Ok(_) => panic!("invalid binding accepted"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn caller_snapshot_nan_payload_is_retained_without_copying_storage() {
    let bits = 0xfff8_0000_0000_0042;
    let tensor = Tensor::new(vec![f64::from_bits(bits), -0.0], vec![1, 2]).unwrap();
    let (function, input) = fixture(Value::Tensor(tensor.clone()));
    let state =
        NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
    let Value::Tensor(actual) = initial_value(&state, 0).unwrap() else {
        panic!("tensor value");
    };
    assert!(actual
        .host_buffer()
        .shares_allocation_with(tensor.host_buffer()));
    assert_eq!(actual.as_f64_slice().unwrap()[0].to_bits(), bits);
    assert_eq!(
        actual.as_f64_slice().unwrap()[1].to_bits(),
        (-0.0_f64).to_bits()
    );
    let snapshot = snapshot_values(&state);
    let Value::Tensor(published) = &snapshot["missing"] else {
        panic!("published tensor");
    };
    assert!(published
        .host_buffer()
        .shares_allocation_with(tensor.host_buffer()));
}

#[test]
fn assigned_binding_requires_an_authoritative_snapshot_value() {
    let (function, mut input) = fixture(Value::Num(1.0));
    input = NativeWorkspaceInput::from_test_values(
        BTreeMap::new(),
        input.local_names,
        input.bindings,
        input.global_bindings,
    );
    assert!(error(&function, input).contains("missing from the caller snapshot"));
}

#[test]
fn duplicate_bindings_remain_rejected() {
    let (function, mut input) = fixture(Value::Num(1.0));
    input.bindings.push(input.bindings[0].clone());
    assert!(error(&function, input).contains("duplicate binding"));
}

#[test]
fn semantic_global_input_becomes_typed_frame_alias_and_completion_metadata() {
    let (function, mut input) = fixture(Value::Num(9.0));
    input.global_bindings.push(NativeWorkspaceGlobalBinding {
        binding: input.bindings[0].binding,
        name: input.bindings[0].name.clone(),
        target: runmat_runtime::workspace::GlobalBindingTarget::Named(
            runmat_runtime::workspace::WorkspaceName::unbounded("session_target"),
        ),
    });
    input = NativeWorkspaceInput::from_test_values(
        BTreeMap::from([("missing".into(), Value::Num(9.0))]),
        input.local_names,
        input.bindings,
        input.global_bindings,
    );
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let authority = runtime.session_variable_authority();
    authority
        .store_global_with_context(&runtime, "session_target", Value::Num(9.0))
        .unwrap();
    let aliases = input.frame_global_bindings(&function, &runtime).unwrap();
    let state = NativeWorkspaceState::new_for_context(
        &function,
        input,
        aliases,
        None,
        Some(authority),
        &runtime,
    )
    .unwrap();
    assert_eq!(
        state
            .service()
            .try_snapshot_with_context(&runtime)
            .unwrap()
            .into_iter()
            .find(|(name, _)| name == "missing")
            .map(|(_, value)| value),
        Some(Value::Num(9.0))
    );
    assert_eq!(
        state.global_aliases.binding_handle_for_slot(0),
        Some(runmat_runtime::workspace::GlobalBindingTarget::Named(
            runmat_runtime::workspace::WorkspaceName::unbounded("session_target"),
        ))
    );
}

#[test]
fn malformed_semantic_global_input_is_rejected_before_execution() {
    let (function, mut input) = fixture(Value::Num(9.0));
    input.global_bindings.push(NativeWorkspaceGlobalBinding {
        binding: BindingId(99),
        name: "missing".into(),
        target: runmat_runtime::workspace::GlobalBindingTarget::LegacySlot(3),
    });
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    assert!(input
        .frame_global_bindings(&function, &runtime)
        .unwrap_err()
        .to_string()
        .contains("is unavailable"));
}

#[test]
fn detached_global_metadata_cannot_retarget_the_caller_generation() {
    let (function, mut input) = fixture(Value::Num(9.0));
    input.global_bindings.push(NativeWorkspaceGlobalBinding {
        binding: BindingId(0),
        name: "missing".into(),
        target: runmat_runtime::workspace::GlobalBindingTarget::LegacySlot(3),
    });
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let aliases = input.frame_global_bindings(&function, &runtime).unwrap();
    let error = NativeWorkspaceState::new_for_context(
        &function,
        input,
        aliases,
        None,
        Some(runtime.session_variable_authority()),
        &runtime,
    )
    .err()
    .expect("detached metadata must be rejected");
    assert!(error.to_string().contains("caller generation target"));
}

#[test]
fn missing_or_mismatched_local_identity_remains_rejected() {
    let (function, mut input) = fixture(Value::Num(1.0));
    input.local_names.clear();
    assert!(error(&function, input).contains("does not match Native IR"));
    let (function, mut input) = fixture(Value::Num(1.0));
    input.local_names.insert(BindingId(0), "another".into());
    assert!(error(&function, input).contains("does not match Native IR"));
    let (function, mut input) = fixture(Value::Num(1.0));
    input.bindings[0].binding = BindingId(99);
    assert!(error(&function, input).contains("is unavailable"));
    let (mut function, input) = fixture(Value::Num(1.0));
    function.locals[0].kind = NativeLocalKind::Temporary;
    assert!(error(&function, input).contains("does not match Native IR"));
}

#[test]
fn unassigned_local_may_be_absent_and_unreferenced_caller_values_are_retained() {
    let (function, mut input) = fixture(Value::Num(1.0));
    input.bindings.clear();
    input = NativeWorkspaceInput::from_test_values(
        BTreeMap::from([("ambient".into(), Value::Num(2.0))]),
        input.local_names,
        input.bindings,
        input.global_bindings,
    );
    let state =
        NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
    assert!(initial_value(&state, 0).is_none());
    assert_eq!(snapshot_values(&state)["ambient"], Value::Num(2.0));
}

#[test]
fn dynamic_assignment_uses_exact_named_and_legacy_targets_before_clear_unbinds_frame() {
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    futures::executor::block_on(runtime.scope(async {
        let (function, mut input) = fixture(Value::Num(7.0));
        input.global_bindings.push(NativeWorkspaceGlobalBinding {
            binding: BindingId(0),
            name: "missing".into(),
            target: runmat_runtime::workspace::GlobalBindingTarget::Named(
                runmat_runtime::workspace::WorkspaceName::unbounded("h"),
            ),
        });
        input = NativeWorkspaceInput::from_test_values(
            BTreeMap::from([("missing".into(), Value::Num(7.0))]),
            input.local_names,
            input.bindings,
            input.global_bindings,
        );
        let aliases = input.frame_global_bindings(&function, &runtime).unwrap();
        let authority = runtime.session_variable_authority();
        authority
            .store_global_with_context(&runtime, "h", Value::Num(7.0))
            .expect("store first named target");
        authority
            .store_global_with_context(&runtime, "j", Value::Num(70.0))
            .expect("store second named target");
        authority
            .store_global_with_context(&runtime, "var_9", Value::Num(90.0))
            .expect("store disjoint named target");
        authority
            .store_global_slot_with_context(&runtime, 9, Value::Num(9.0))
            .expect("store legacy slot target");
        let state = NativeWorkspaceState::new_for_context(
            &function,
            input,
            aliases.clone(),
            None,
            Some(authority.clone()),
            &runtime,
        )
        .unwrap();
        let service = state.service();
        service
            .assign_with_context(&runtime, "missing", Value::Num(8.0))
            .unwrap();
        assert_eq!(authority.global_value("h").unwrap(), Some(Value::Num(8.0)),);
        service
            .bind_base_global_target_with_context(
                &runtime,
                "missing",
                &runmat_runtime::workspace::GlobalBindingTarget::Named(
                    runmat_runtime::workspace::WorkspaceName::unbounded("j"),
                ),
            )
            .unwrap();
        service
            .assign_with_context(&runtime, "missing", Value::Num(10.0))
            .unwrap();
        assert_eq!(authority.global_value("h").unwrap(), Some(Value::Num(8.0)),);
        assert_eq!(authority.global_value("j").unwrap(), Some(Value::Num(10.0)));
        service
            .bind_base_global_target_with_context(
                &runtime,
                "missing",
                &runmat_runtime::workspace::GlobalBindingTarget::LegacySlot(9),
            )
            .unwrap();
        service
            .assign_with_context(&runtime, "missing", Value::Num(11.0))
            .unwrap();
        assert_eq!(authority.global_slot(9).unwrap(), Some(Value::Num(11.0)));
        assert_eq!(
            state.global_aliases.binding_handle_for_slot(0),
            Some(runmat_runtime::workspace::GlobalBindingTarget::LegacySlot(
                9
            )),
        );
        assert_eq!(
            authority.global_value("var_9").unwrap(),
            Some(Value::Num(90.0)),
        );
        service.remove_with_context(&runtime, "missing").unwrap();
        assert!(!aliases.has_binding_for_slot(0));
        service
            .assign_with_context(&runtime, "missing", Value::Num(12.0))
            .unwrap();
        assert_eq!(authority.global_slot(9).unwrap(), Some(Value::Num(11.0)));
        service.clear_with_context(&runtime).unwrap();
        assert!(service
            .try_snapshot_with_context(&runtime)
            .unwrap()
            .is_empty());
        assert_eq!(authority.global_slot(9).unwrap(), Some(Value::Num(11.0)));
    }));
}

#[test]
fn admitted_native_write_remains_publishable_after_execution_state_drops() {
    let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let mut initial = runmat_runtime::workspace::WorkspaceDelta::new();
    initial.assign("missing", Value::Num(1.0)).unwrap();
    owner.apply(initial).unwrap();
    let projection = std::rc::Rc::new(
        runmat_runtime::workspace::BaseWorkspaceProjectionService::new(
            owner.begin_projection().unwrap(),
        ),
    );
    let canonical: std::rc::Rc<dyn runmat_runtime::context::RuntimeWorkspaceService> =
        projection.clone();
    let (function, input) = fixture(Value::Num(1.0));
    let state =
        NativeWorkspaceState::new(&function, input, Default::default(), Some(canonical), None)
            .unwrap();
    assert!(!projection.has_pending_changes().unwrap());
    state.synchronize_local(0, Value::Num(42.0)).unwrap();
    drop(state);

    assert!(projection.has_pending_changes().unwrap());
    projection.publish().unwrap();
    assert_eq!(owner.lookup("missing").unwrap(), Some(Value::Num(42.0)));
}

#[test]
fn context_bound_native_workspace_forwards_the_exact_operation() {
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let mut initial = runmat_runtime::workspace::WorkspaceDelta::new();
    initial.assign("missing", Value::Num(1.0)).unwrap();
    owner.apply(initial).unwrap();
    let projection = std::rc::Rc::new(
        runmat_runtime::workspace::BaseWorkspaceProjectionService::new_for_context(
            owner.begin_projection().unwrap(),
            &runtime,
        ),
    );
    let canonical: std::rc::Rc<dyn runmat_runtime::context::RuntimeWorkspaceService> =
        projection.clone();
    let (function, input) = fixture(Value::Num(1.0));
    let state = NativeWorkspaceState::new_for_context(
        &function,
        input,
        Default::default(),
        Some(canonical),
        None,
        &runtime,
    )
    .unwrap();
    let service = state.service();
    let unrelated = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    assert_eq!(
        service
            .ensure_mutation_authority_with_context(&unrelated)
            .unwrap_err()
            .identifier(),
        Some("RunMat:Workspace:MutationContextMismatch")
    );
    assert_eq!(
        service
            .try_lookup_with_context(&unrelated, "missing")
            .unwrap_err()
            .identifier(),
        Some("RunMat:Workspace:MutationContextMismatch")
    );
    assert_eq!(
        service.try_lookup("missing").unwrap_err().identifier(),
        Some("RunMat:Workspace:PublicationContextRequired")
    );
    assert_eq!(
        service
            .try_lookup_with_context(&runtime, "missing")
            .unwrap(),
        Some(Value::Num(1.0))
    );
    service
        .ensure_mutation_authority_with_context(&runtime)
        .unwrap();
    state
        .synchronize_local_with_context(&runtime, 0, Value::Num(3.0))
        .unwrap();
    assert!(projection.has_pending_changes().unwrap());
}

#[test]
fn context_bound_native_workspace_rejects_unrelated_context_without_delegate() {
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let unrelated = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let (function, input) = fixture(Value::Num(1.0));
    let state = NativeWorkspaceState::new_for_context(
        &function,
        input,
        Default::default(),
        None,
        None,
        &runtime,
    )
    .unwrap();
    let service = state.service();
    assert_eq!(
        service
            .try_lookup_with_context(&unrelated, "missing")
            .unwrap_err()
            .identifier(),
        Some("RunMat:Workspace:MutationContextMismatch")
    );
    assert_eq!(
        service
            .assign_with_context(&unrelated, "missing", Value::Num(2.0))
            .unwrap_err()
            .identifier(),
        Some("RunMat:Workspace:MutationContextMismatch")
    );
    assert_eq!(
        service
            .try_lookup_with_context(&runtime, "missing")
            .unwrap(),
        Some(Value::Num(1.0))
    );
}

#[test]
fn retained_native_workspace_service_rejects_effects_after_executor_drop() {
    let (function, input) = fixture(Value::Num(1.0));
    let state =
        NativeWorkspaceState::new(&function, input, Default::default(), None, None).unwrap();
    let service = state.service();
    drop(state);

    for error in [
        service.assign("missing", Value::Num(2.0)).unwrap_err(),
        service.remove("missing").unwrap_err(),
        service.clear().unwrap_err(),
    ] {
        assert_eq!(error.identifier(), Some("RunMat:Workspace:FrameClosed"));
    }
    assert!(service.try_snapshot().is_err());
    assert_eq!(service.lookup("missing"), None);
}

#[test]
fn active_native_checked_reads_preserve_authority_and_borrow_failures() {
    let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let canonical: std::rc::Rc<dyn runmat_runtime::context::RuntimeWorkspaceService> =
        std::rc::Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            owner.clone(),
        ));
    let (function, input) = fixture(Value::Num(1.0));
    let state =
        NativeWorkspaceState::new(&function, input, Default::default(), Some(canonical), None)
            .unwrap();
    let service = state.service();

    {
        let _borrow = state.frame.lexical.borrow_mut();
        let error = service.try_lookup("missing").unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NativeWorkspaceBorrow"));
        let error = service.try_snapshot().unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:NativeWorkspaceBorrow"));
    }

    owner.close().unwrap();
    for error in [
        service.try_lookup("missing").unwrap_err(),
        service.try_snapshot().unwrap_err(),
        service.try_global_names().unwrap_err(),
    ] {
        assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    }
}

#[test]
fn internal_native_stores_check_authority_for_global_and_function_frames() {
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let canonical: std::rc::Rc<dyn runmat_runtime::context::RuntimeWorkspaceService> =
        std::rc::Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            owner.clone(),
        ));
    let (function, mut input) = fixture(Value::Num(1.0));
    let target = runmat_runtime::workspace::GlobalBindingTarget::Named(
        runmat_runtime::workspace::WorkspaceName::unbounded("session_target"),
    );
    input.global_bindings.push(NativeWorkspaceGlobalBinding {
        binding: BindingId(0),
        name: "missing".into(),
        target,
    });
    input = NativeWorkspaceInput::from_test_values(
        BTreeMap::from([("missing".into(), Value::Num(1.0))]),
        input.local_names,
        input.bindings,
        input.global_bindings,
    );
    let retained_initial = input.generation.read_lease("missing").unwrap();
    let authority = runtime.session_variable_authority();
    authority
        .store_global_with_context(&runtime, "session_target", Value::Num(1.0))
        .unwrap();
    let aliases = input.frame_global_bindings(&function, &runtime).unwrap();
    let state = NativeWorkspaceState::new_for_context(
        &function,
        input,
        aliases,
        Some(canonical.clone()),
        Some(authority.clone()),
        &runtime,
    )
    .unwrap();
    let function_state = NativeWorkspaceState::function_frame_for_context(
        &function,
        Default::default(),
        Some(canonical),
        Some(authority),
        &runtime,
    )
    .unwrap();
    assert_eq!(
        state
            .service()
            .try_lookup_with_context(&runtime, "missing")
            .unwrap(),
        Some(Value::Num(1.0))
    );
    assert_eq!(
        function_state
            .service()
            .try_lookup_with_context(&runtime, "missing")
            .unwrap(),
        None
    );
    owner.close().unwrap();

    for error in [
        state
            .synchronize_local_with_context(&runtime, 0, Value::Num(2.0))
            .unwrap_err(),
        function_state
            .synchronize_local_with_context(&runtime, 0, Value::Num(3.0))
            .unwrap_err(),
    ] {
        let crate::NativeExecutorError::Runtime(error) = error else {
            panic!("expected typed Runtime workspace rejection")
        };
        assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    }
    assert_eq!(retained_initial.value(), &Value::Num(1.0));
}

#[test]
fn procedure_workspace_isolated_from_live_parent_but_inherits_revocation() {
    let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let mut parent = runmat_runtime::workspace::WorkspaceDelta::new();
    parent.assign("missing", Value::Num(1.0)).unwrap();
    owner.apply(parent).unwrap();
    let canonical: std::rc::Rc<dyn runmat_runtime::context::RuntimeWorkspaceService> =
        std::rc::Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            owner.clone(),
        ));
    let (function, _) = fixture(Value::Num(0.0));
    let state =
        NativeWorkspaceState::function_frame(&function, Default::default(), Some(canonical), None);
    state.initialize_local(0, Value::Num(2.0)).unwrap();
    let service = state.service();

    assert_eq!(
        service.try_lookup("missing").unwrap(),
        Some(Value::Num(2.0))
    );
    service.assign("missing", Value::Num(3.0)).unwrap();
    assert_eq!(
        service.try_lookup("missing").unwrap(),
        Some(Value::Num(3.0))
    );
    assert_eq!(owner.lookup("missing").unwrap(), Some(Value::Num(1.0)));
    service.clear().unwrap();
    assert!(service.try_snapshot().unwrap().is_empty());
    assert_eq!(owner.lookup("missing").unwrap(), Some(Value::Num(1.0)));

    owner.close().unwrap();
    let error = service.assign("missing", Value::Num(4.0)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
}
