use futures::executor::block_on;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
};
use runmat_value::{CellArray, Value};

const FORMS: &[(&str, usize)] = &[
    ("ldivide", 2),
    ("minus", 2),
    ("plus", 2),
    ("power", 2),
    ("rdivide", 2),
    ("times", 2),
    ("conj", 1),
    ("imag", 1),
    ("real", 1),
    ("complex", 1),
    ("complex", 2),
];

#[test]
fn excess_outputs_reject_before_resident_operand_access() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 3], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    assert!(!runmat_accelerate_api::registered_providers()
        .iter()
        .any(|provider| provider.device_id() == allocation.handle.device_id()));
    let caller = allocation.handle.clone();
    let descriptor = caller.wire_descriptor();
    let unavailable = Value::GpuTensor(allocation.handle);
    for &(name, inputs) in FORMS {
        let entry = runmat_builtins::builtin_catalog_entry_by_name(name).unwrap();
        let identifier = format!("RunMat:{name}:TooManyOutputs");
        assert!(entry
            .descriptor
            .errors
            .iter()
            .any(|error| error.identifier == Some(identifier.as_str())));
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            name,
            &vec![unavailable.clone(); inputs],
            2,
        ))
        .expect_err("output admission must precede provider lookup/download");
        assert_eq!(error.identifier(), Some(identifier.as_str()), "{name}");
        assert_eq!(caller.wire_descriptor(), descriptor);
        assert_eq!(allocation.probe.retirements(), 0);
    }
    drop(unavailable);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(caller);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn zero_outputs_still_execute_input_admission() {
    for &(name, inputs) in FORMS {
        assert!(
            block_on(runmat_runtime::call_builtin_async_with_outputs(
                name,
                &vec![Value::Num(2.0); inputs],
                0,
            ))
            .is_ok(),
            "{name}: zero output invocation remains legal"
        );
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            name,
            &vec![Value::Cell(CellArray::new(vec![Value::Num(2.0)], 1, 1).unwrap()); inputs],
            0,
        ))
        .expect_err("discarding results must not discard input validation");
        assert_ne!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str())
        );
    }
}
