use super::*;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{IntValue, Tensor};

#[test]
fn moves_fixed_width_payloads_without_conversion() {
    let input = structure(&[
        ("wide", Value::Int(IntValue::U64(u64::MAX))),
        ("small", Value::Int(IntValue::I8(-7))),
    ]);
    let Value::Struct(output) = call(Value::Struct(input), Vec::new()).unwrap() else {
        panic!("expected structure")
    };
    assert_eq!(output.fields["wide"], Value::Int(IntValue::U64(u64::MAX)));
    assert_eq!(output.fields["small"], Value::Int(IntValue::I8(-7)));
}

#[test]
fn nested_resident_handle_remains_resident() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![42.0], vec![1, 1]).unwrap();
        let values = tensor.materialize_f64();
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &tensor.shape,
            })
            .unwrap();
        let input = structure(&[("z", Value::GpuTensor(handle.clone())), ("a", 1.0.into())]);
        let Value::Struct(output) = call(Value::Struct(input), Vec::new()).unwrap() else {
            panic!("expected structure")
        };
        assert_eq!(output.fields["z"], Value::GpuTensor(handle));
    });
}
