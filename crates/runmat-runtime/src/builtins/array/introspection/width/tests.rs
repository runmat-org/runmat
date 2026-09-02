use super::*;
use futures::executor::block_on;
use runmat_value::{IntegerStorage, Tensor, Value};

fn run(value: Value) -> crate::BuiltinResult<Value> {
    block_on(width_builtin(value))
}

#[test]
fn returns_second_visible_dimension_for_arrays() {
    assert_eq!(run(Value::Num(4.0)).unwrap(), Value::Num(1.0));
    assert_eq!(
        run(Value::Tensor(
            Tensor::new(vec![0.0; 15], vec![5, 3]).unwrap()
        ))
        .unwrap(),
        Value::Num(3.0)
    );
    assert_eq!(
        run(Value::Tensor(Tensor::new(Vec::new(), vec![0, 7]).unwrap())).unwrap(),
        Value::Num(7.0)
    );
}

#[test]
fn every_integer_class_uses_shape_without_reading_values() {
    for storage in [
        IntegerStorage::I8(vec![1; 6]),
        IntegerStorage::I16(vec![1; 6]),
        IntegerStorage::I32(vec![1; 6]),
        IntegerStorage::I64(vec![1; 6]),
        IntegerStorage::U8(vec![1; 6]),
        IntegerStorage::U16(vec![1; 6]),
        IntegerStorage::U32(vec![1; 6]),
        IntegerStorage::U64(vec![u64::MAX; 6]),
    ] {
        let tensor = Tensor::new_integer(storage, vec![2, 3]).unwrap();
        assert_eq!(run(Value::Tensor(tensor)).unwrap(), Value::Num(3.0));
    }
}

#[test]
fn table_width_counts_variables_not_payload_columns() {
    let table = crate::builtins::table::table_from_columns(
        vec!["samples".into(), "label".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0; 6], vec![3, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0; 3], vec![3, 1]).unwrap()),
        ],
    )
    .unwrap();
    assert_eq!(run(table).unwrap(), Value::Num(2.0));
}

#[test]
fn resident_shape_requires_no_provider_or_payload_access() {
    let handle = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![4, 13],
        device_id: u32::MAX,
        buffer_id: u64::MAX,
        descriptor: Default::default(),
    };
    assert_eq!(run(Value::GpuTensor(handle)).unwrap(), Value::Num(13.0));
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = run(Value::Num(1.0)).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:width:TooManyOutputs"));
}
