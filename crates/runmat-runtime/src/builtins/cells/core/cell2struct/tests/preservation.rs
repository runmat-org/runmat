use super::*;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{IntValue, Tensor};

#[test]
fn exact_integer_payloads_move_without_conversion() {
    for payload in [
        IntValue::I8(i8::MIN),
        IntValue::I16(i16::MIN),
        IntValue::I32(i32::MIN),
        IntValue::I64(i64::MIN),
        IntValue::U8(u8::MAX),
        IntValue::U16(u16::MAX),
        IntValue::U32(u32::MAX),
        IntValue::U64(u64::MAX),
    ] {
        let cells = CellArray::new(vec![Value::Int(payload.clone())], 1, 1).unwrap();
        let Value::Struct(output) = call(cells, Value::from("value"), None).unwrap() else {
            panic!("expected structure")
        };
        assert_eq!(field(&output, "value"), &Value::Int(payload));
    }
}

#[test]
fn nested_resident_handle_moves_without_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![7.0], vec![1, 1]).unwrap();
        let values = tensor.materialize_f64();
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &tensor.shape,
            })
            .unwrap();
        let cells = CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).unwrap();
        let Value::Struct(output) = call(cells, Value::from("resident"), None).unwrap() else {
            panic!("expected structure")
        };
        assert_eq!(field(&output, "resident"), &Value::GpuTensor(handle));
    });
}
