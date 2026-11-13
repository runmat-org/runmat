use super::*;

// Terminal test projection: this conversion publishes only numerical or
// resident GPU values, never a GC-managed object graph.
async fn im2double_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::im2double_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
    .map(|owned| owned.with_owned(|value| value))
}
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
use runmat_value::{IntegerStorage, LogicalArray, NumericDType};

fn call(value: Value) -> Value {
    block_on(im2double_builtin(value, Vec::new())).expect("im2double")
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_uint8_images_keep_conversion_values_and_scalarization() {
    let scalar = Tensor::new_integer(IntegerStorage::U8(vec![255]), Vec::new()).unwrap();
    assert_eq!(call(Value::Tensor(scalar)), Value::Num(1.0));
    let vector = Tensor::new_integer(IntegerStorage::U8(vec![0, 255]), vec![2]).unwrap();
    let Value::Tensor(output) = call(Value::Tensor(vector)) else {
        panic!("two-element image must remain a tensor")
    };
    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.numeric_dtype(), NumericDType::F64);
    assert_eq!(output.materialize_f64(), vec![0.0, 1.0]);
}

#[test]
fn scales_uint8_tensor_to_unit_double() {
    let input =
        Tensor::new_with_dtype(vec![0.0, 128.0, 255.0], vec![1, 3], NumericDType::U8).unwrap();
    let Value::Tensor(out) = call(Value::Tensor(input)) else {
        panic!("expected tensor");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    let values = values(&out);
    assert_eq!(values[0], 0.0);
    assert!((values[1] - 128.0 / 255.0).abs() < 1e-12);
    assert_eq!(values[2], 1.0);
}

#[test]
fn im2double_reads_typed_integer_tensor_storage_exactly() {
    let input = Tensor::new_integer(IntegerStorage::U8(vec![0, 128, 255]), vec![1, 3]).unwrap();

    let Value::Tensor(out) = call(Value::Tensor(input)) else {
        panic!("expected tensor");
    };

    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    let values = values(&out);
    assert_eq!(values[0], 0.0);
    assert!((values[1] - 128.0 / 255.0).abs() < 1e-12);
    assert_eq!(values[2], 1.0);
}

#[test]
fn scales_uint16_scalar() {
    assert_eq!(call(Value::Int(IntValue::U16(65535))), Value::Num(1.0));
}

#[test]
fn scales_int16_image_range() {
    assert_eq!(call(Value::Int(IntValue::I16(i16::MIN))), Value::Num(0.0));
    assert_eq!(call(Value::Int(IntValue::I16(i16::MAX))), Value::Num(1.0));
    let input =
        Tensor::new_integer(IntegerStorage::I16(vec![i16::MIN, 0, i16::MAX]), vec![1, 3]).unwrap();
    let Value::Tensor(out) = call(Value::Tensor(input)) else {
        panic!("expected tensor");
    };
    let values = values(&out);
    assert_eq!(values[0], 0.0);
    assert!((values[1] - 32768.0 / 65535.0).abs() < 1e-12);
    assert_eq!(values[2], 1.0);
}

#[test]
fn preserves_float_values_and_shape() {
    let input =
        Tensor::new_with_dtype(vec![-0.25, 0.0, 0.5, 1.25], vec![2, 2], NumericDType::F32).unwrap();
    let Value::Tensor(out) = call(Value::Tensor(input)) else {
        panic!("expected tensor");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(values(&out), vec![-0.25, 0.0, 0.5, 1.25]);
}

#[test]
fn converts_logical_array_to_double_zeros_and_ones() {
    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
    let Value::Tensor(out) = call(Value::LogicalArray(logical)) else {
        panic!("expected tensor");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(values(&out), vec![1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn rejects_unsupported_integer_classes() {
    for (scalar, storage) in [
        (IntValue::I8(0), IntegerStorage::I8(vec![0])),
        (IntValue::I32(0), IntegerStorage::I32(vec![0])),
        (IntValue::I64(0), IntegerStorage::I64(vec![0])),
        (IntValue::U32(0), IntegerStorage::U32(vec![0])),
        (IntValue::U64(0), IntegerStorage::U64(vec![0])),
    ] {
        let scalar_err = block_on(im2double_builtin(Value::Int(scalar), Vec::new())).unwrap_err();
        assert_eq!(
            scalar_err.identifier(),
            IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
        );
        let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        let tensor_err =
            block_on(im2double_builtin(Value::Tensor(tensor), Vec::new())).unwrap_err();
        assert_eq!(
            tensor_err.identifier(),
            IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
        );
    }
}

#[test]
fn im2double_descriptor_signatures_cover_surface() {
    let labels: Vec<&str> = IM2DOUBLE_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(
        labels,
        vec!["J = im2double(I)", "J = im2double(I, \"indexed\")"]
    );
}

#[test]
fn indexed_mode_adds_one_to_zero_based_integer_indices() {
    let input = Tensor::new_integer(IntegerStorage::U16(vec![0, 7, u16::MAX]), vec![1, 3])
        .expect("indexed uint16 image");
    let Value::Tensor(out) = block_on(im2double_builtin(
        Value::Tensor(input),
        vec![Value::String("indexed".into())],
    ))
    .expect("indexed im2double") else {
        panic!("expected tensor");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::F64);
    assert_eq!(out.shape(), &[1, 3]);
    assert_eq!(values(&out), vec![1.0, 8.0, 65536.0]);
}

#[test]
fn indexed_mode_preserves_double_and_logical_values() {
    let Value::Tensor(double_out) = block_on(im2double_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 3.5], vec![1, 2]).unwrap()),
        vec![Value::String("InDeXeD".into())],
    ))
    .expect("indexed double") else {
        panic!("expected tensor");
    };
    assert_eq!(values(&double_out), vec![1.0, 3.5]);

    let Value::Tensor(logical_out) = block_on(im2double_builtin(
        Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
        vec![Value::String("indexed".into())],
    ))
    .expect("indexed logical") else {
        panic!("expected tensor");
    };
    assert_eq!(values(&logical_out), vec![0.0, 1.0]);
}

#[test]
fn indexed_mode_rejects_int16_and_single_inputs() {
    for input in [
        Tensor::new_integer(IntegerStorage::I16(vec![0]), vec![1, 1]).unwrap(),
        Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap(),
    ] {
        let err = block_on(im2double_builtin(
            Value::Tensor(input),
            vec![Value::String("indexed".into())],
        ))
        .expect_err("unsupported indexed class");
        assert_eq!(
            err.identifier(),
            IM2DOUBLE_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
        );
    }
}

#[test]
fn im2double_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = IM2DOUBLE_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.IM2DOUBLE.TOO_MANY_INPUTS"));
    assert!(codes.contains(&"RM.IM2DOUBLE.TOO_MANY_OUTPUTS"));
    assert!(codes.contains(&"RM.IM2DOUBLE.INVALID_INPUT"));
    assert!(codes.contains(&"RM.IM2DOUBLE.UNSUPPORTED_INPUT_TYPE"));
    assert!(codes.contains(&"RM.IM2DOUBLE.INTERNAL"));
}

#[test]
fn im2double_too_many_args_uses_stable_identifier() {
    let err = block_on(im2double_builtin(
        Value::Num(1.0),
        vec![Value::from("indexed"), Value::Num(2.0)],
    ))
    .expect_err("expected argument error");
    assert_eq!(err.identifier(), IM2DOUBLE_ERROR_TOO_MANY_INPUTS.identifier);
}

#[test]
fn im2double_resident_integer_input_restores_double_output_to_owner() {
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&[0, 128, 255]),
                shape: &[1, 3],
            })
            .expect("upload uint8 image");
        let Value::GpuTensor(output) =
            block_on(im2double_builtin(Value::GpuTensor(input), Vec::new()))
                .expect("resident im2double")
        else {
            panic!("expected resident double output");
        };
        assert_eq!(output.device_id(), provider.device_id());
        assert_eq!(runmat_accelerate_api::handle_integer_type(&output), None);
        let host = block_on(provider.download(&output)).expect("download double image");
        assert_eq!(host.shape, vec![1, 3]);
        assert_eq!(host.data, vec![0.0, 128.0 / 255.0, 1.0]);
    });
}

#[test]
fn empty_nd_image_preserves_geometry_and_output_class() {
    let image = Tensor::new_integer(IntegerStorage::U8(vec![]), vec![0, 2, 3]).unwrap();
    let Value::Tensor(output) = call(Value::Tensor(image)) else {
        panic!("empty image must remain a tensor");
    };
    assert_eq!(output.shape(), &[0, 2, 3]);
    assert_eq!(output.numeric_dtype(), NumericDType::F64);
    assert!(values(&output).is_empty());
}

#[test]
fn invalid_mode_rejects_before_resident_download() {
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&[0, 255]),
                shape: &[1, 2],
            })
            .unwrap();
        let err = block_on(im2double_builtin(
            Value::GpuTensor(input),
            vec![Value::String("not-indexed".into())],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), IM2DOUBLE_ERROR_INVALID_INPUT.identifier);
    });
}
