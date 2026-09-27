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
    let state = NativeWorkspaceState::new(&function, input).unwrap();
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
    match NativeWorkspaceState::new(function, input) {
        Ok(_) => panic!("invalid binding accepted"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn caller_snapshot_nan_payload_is_retained_without_copying_storage() {
    let bits = 0xfff8_0000_0000_0042;
    let tensor = Tensor::new(vec![f64::from_bits(bits), -0.0], vec![1, 2]).unwrap();
    let (function, input) = fixture(Value::Tensor(tensor.clone()));
    let state = NativeWorkspaceState::new(&function, input).unwrap();
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
    let state = NativeWorkspaceState::new(&function, input).unwrap();
    assert!(state.initial_value(0).is_none());
    assert_eq!(state.snapshot().unwrap().values["ambient"], Value::Num(2.0));
}
