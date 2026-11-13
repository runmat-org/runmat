use super::*;
use futures::executor::block_on;
use runmat_value::{CellArray, IntValue, ObjectInstance};

#[test]
fn backing_fields_preserve_wide_integer_and_nested_resident_payloads() {
    let source = Value::Object(ObjectInstance::new("PropertyCutover"));
    assert_eq!(
        block_on(get_p_builtin(source.clone())).unwrap(),
        Value::Num(0.0)
    );
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![0, 3],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let payload = Value::Cell(
        CellArray::new(
            vec![
                Value::Int(IntValue::U64(u64::MAX)),
                Value::GpuTensor(allocation.handle),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let updated = block_on(set_p_builtin(source.clone(), payload.clone())).unwrap();
    assert_eq!(block_on(get_p_builtin(updated)).unwrap(), payload);
    assert_eq!(block_on(get_p_builtin(source)).unwrap(), Value::Num(0.0));
    assert_eq!(allocation.probe.retirements(), 0);
    drop(payload);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn output_bounds_precede_receiver_validation() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    assert_eq!(
        block_on(get_p_builtin(Value::Num(1.0)))
            .unwrap_err()
            .identifier(),
        Some("RunMat:get.p:TooManyOutputs")
    );
    assert_eq!(
        block_on(set_p_builtin(Value::Num(1.0), Value::Num(2.0)))
            .unwrap_err()
            .identifier(),
        Some("RunMat:set.p:TooManyOutputs")
    );
}
