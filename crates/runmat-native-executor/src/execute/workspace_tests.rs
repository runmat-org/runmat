use super::{NativeWorkspaceBinding, NativeWorkspaceInput, NativeWorkspaceState};
use runmat_native_codegen::{
    NativeBlockId, NativeFunction, NativeFunctionAbi, NativeLocalId, NativeLocalKind,
    NativeLocalMetadata,
};
use runmat_types::{BindingId, ProgramFunctionId, ProgramSourceId};
use runmat_value::{Tensor, Value};
use std::collections::BTreeMap;

#[test]
fn only_external_workspace_mutation_requests_reconciliation() {
    let (function, input) = fixture(Value::Num(1.0));
    let state = NativeWorkspaceState::new(&function, input, Default::default(), None).unwrap();
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

fn fixture(value: Value) -> (NativeFunction, NativeWorkspaceInput) {
    let binding = BindingId(0);
    let function = NativeFunction {
        id: ProgramFunctionId(0),
        source: ProgramSourceId(0),
        name: "snapshot".into(),
        source_role: runmat_types::FunctionSourceRole::Unknown,
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
        locals: vec![NativeLocalMetadata {
            id: NativeLocalId(0),
            binding: Some(binding),
            name: Some("missing".into()),
            kind: NativeLocalKind::Binding,
        }],
        entry: NativeBlockId(0),
        blocks: vec![],
        expected_sites: vec![],
    };
    let input = NativeWorkspaceInput {
        values: BTreeMap::from([("missing".into(), value)]),
        local_names: BTreeMap::from([(binding, "missing".into())]),
        bindings: vec![NativeWorkspaceBinding {
            binding,
            name: "missing".into(),
        }],
    };
    (function, input)
}

fn error(function: &NativeFunction, input: NativeWorkspaceInput) -> String {
    match NativeWorkspaceState::new(function, input, Default::default(), None) {
        Ok(_) => panic!("invalid binding accepted"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn caller_snapshot_nan_payload_is_retained_without_copying_storage() {
    let bits = 0xfff8_0000_0000_0042;
    let tensor = Tensor::new(vec![f64::from_bits(bits), -0.0], vec![1, 2]).unwrap();
    let (function, input) = fixture(Value::Tensor(tensor.clone()));
    let state = NativeWorkspaceState::new(&function, input, Default::default(), None).unwrap();
    let Value::Tensor(actual) = state.initial_value(0).unwrap() else {
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
    let snapshot = state.snapshot().unwrap();
    let Value::Tensor(published) = &snapshot.values["missing"] else {
        panic!("published tensor");
    };
    assert!(published
        .host_buffer()
        .shares_allocation_with(tensor.host_buffer()));
}

#[test]
fn assigned_binding_requires_an_authoritative_snapshot_value() {
    let (function, mut input) = fixture(Value::Num(1.0));
    input.values.clear();
    assert!(error(&function, input).contains("missing from the caller snapshot"));
}

#[test]
fn duplicate_bindings_remain_rejected() {
    let (function, mut input) = fixture(Value::Num(1.0));
    input.bindings.push(input.bindings[0].clone());
    assert!(error(&function, input).contains("duplicate binding"));
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
    input.values.clear();
    input.values.insert("ambient".into(), Value::Num(2.0));
    let state = NativeWorkspaceState::new(&function, input, Default::default(), None).unwrap();
    assert!(state.initial_value(0).is_none());
    assert_eq!(state.snapshot().unwrap().values["ambient"], Value::Num(2.0));
}

#[test]
fn dynamic_assignment_publishes_active_global_and_clear_unbinds_only_the_frame() {
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    futures::executor::block_on(runtime.scope(async {
        let (function, input) = fixture(Value::Num(7.0));
        let aliases = runmat_runtime::workspace::bindings::FrameGlobalBindings::default();
        aliases.declare_named(0, "missing".into());
        runmat_runtime::workspace::session::store_global_named("missing", Value::Num(7.0));
        let state = NativeWorkspaceState::new(&function, input, aliases.clone(), None).unwrap();
        let service = state.service();
        service.assign("missing", Value::Num(8.0)).unwrap();
        assert_eq!(
            runmat_runtime::workspace::session::global_value("missing"),
            Some(Value::Num(8.0))
        );
        service.remove("missing").unwrap();
        assert_eq!(aliases.name_for_slot(0), None);
        service.assign("missing", Value::Num(9.0)).unwrap();
        assert_eq!(
            runmat_runtime::workspace::session::global_value("missing"),
            Some(Value::Num(8.0))
        );
        service.clear().unwrap();
        assert_eq!(state.snapshot().unwrap().values.len(), 0);
        assert_eq!(
            runmat_runtime::workspace::session::global_value("missing"),
            Some(Value::Num(8.0))
        );
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
        NativeWorkspaceState::new(&function, input, Default::default(), Some(canonical)).unwrap();
    assert!(!projection.has_pending_changes().unwrap());
    state.synchronize_local(0, Value::Num(42.0)).unwrap();
    drop(state);

    assert!(projection.has_pending_changes().unwrap());
    projection.publish().unwrap();
    assert_eq!(owner.lookup("missing").unwrap(), Some(Value::Num(42.0)));
}

#[test]
fn retained_native_workspace_service_rejects_effects_after_executor_drop() {
    let (function, input) = fixture(Value::Num(1.0));
    let state = NativeWorkspaceState::new(&function, input, Default::default(), None).unwrap();
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
        NativeWorkspaceState::new(&function, input, Default::default(), Some(canonical)).unwrap();
    let service = state.service();

    {
        let _borrow = state.values.borrow_mut();
        let error = service.try_lookup("missing").unwrap_err();
        assert_eq!(error.identifier(), Some("NativeWorkspaceBorrow"));
        let error = service.try_snapshot().unwrap_err();
        assert_eq!(error.identifier(), Some("NativeWorkspaceBorrow"));
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
    let owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let canonical: std::rc::Rc<dyn runmat_runtime::context::RuntimeWorkspaceService> =
        std::rc::Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
            owner.clone(),
        ));
    let (function, input) = fixture(Value::Num(1.0));
    let aliases = runmat_runtime::workspace::bindings::FrameGlobalBindings::default();
    aliases.declare_named(0, "missing".into());
    let state =
        NativeWorkspaceState::new(&function, input, aliases, Some(canonical.clone())).unwrap();
    let function_state =
        NativeWorkspaceState::function_frame(&function, Default::default(), Some(canonical));
    owner.close().unwrap();

    for error in [
        state.synchronize_local(0, Value::Num(2.0)).unwrap_err(),
        function_state
            .synchronize_local(0, Value::Num(3.0))
            .unwrap_err(),
    ] {
        let crate::NativeExecutorError::Runtime(error) = error else {
            panic!("expected typed Runtime workspace rejection")
        };
        assert_eq!(error.identifier(), Some("RunMat:Workspace:Closed"));
    }
    assert_eq!(state.initial_value(0), Some(Value::Num(1.0)));
    assert_eq!(function_state.initial_value(0), None);
}
