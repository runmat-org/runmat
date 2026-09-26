use super::*;
use futures::executor::block_on;
use runmat_value::{CellArray, IntValue, ObjectInstance};

#[test]
fn backing_fields_preserve_wide_integer_and_nested_resident_payloads() {
    let source = Value::Object(ObjectInstance::new("PropertyCutover".into()));
    assert_eq!(
        block_on(get_p_builtin(source.clone())).unwrap(),
        Value::Num(0.0)
    );
    let payload = Value::Cell(
        CellArray::new(
            vec![
                Value::Int(IntValue::U64(u64::MAX)),
                Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle {
                    shape: vec![0, 3],
                    device_id: u32::MAX - 732,
                    buffer_id: u64::MAX - 732,
                    descriptor: Default::default(),
                }),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let updated = block_on(set_p_builtin(source.clone(), payload.clone())).unwrap();
    assert_eq!(block_on(get_p_builtin(updated)).unwrap(), payload);
    assert_eq!(block_on(get_p_builtin(source)).unwrap(), Value::Num(0.0));
}

#[test]
fn output_bounds_precede_receiver_validation() {
    let _guard = crate::output_count::push_output_count(Some(2));
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
