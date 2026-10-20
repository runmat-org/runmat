use super::*;

// Terminal test projection: this conversion publishes only numerical or
// resident GPU values, never a GC-managed object graph.
async fn im2uint16_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    super::im2uint16_builtin(
        None,
        value,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
    .map(|owned| owned.with_owned(|value| value))
}
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::{
    HostIntegerDataOwned, HostIntegerDataView, HostIntegerTensorView, IntegerElementType,
};
use runmat_value::{IntegerStorage, LogicalArray};

fn call(value: Value) -> Value {
    block_on(im2uint16_builtin(value, Vec::new())).expect("im2uint16")
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.materialize_f64()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn low_rank_uint8_images_preserve_uint16_class_and_scalarization() {
    let scalar = Tensor::new_integer(IntegerStorage::U8(vec![255]), Vec::new()).unwrap();
    assert_eq!(
        call(Value::Tensor(scalar)),
        Value::Int(IntValue::U16(u16::MAX))
    );
    let vector = Tensor::new_integer(IntegerStorage::U8(vec![0, 255]), vec![2]).unwrap();
    let Value::Tensor(output) = call(Value::Tensor(vector)) else {
        panic!("two-element image must remain a tensor")
    };
    assert_eq!(output.shape(), &[2]);
    assert_eq!(
        output.integer_storage(),
        Some(&IntegerStorage::U16(vec![0, u16::MAX]))
    );
}

#[test]
fn scales_double_to_uint16_image_range() {
    assert_eq!(call(Value::Num(1.0)), Value::Int(IntValue::U16(65535)));
}

#[test]
fn expands_uint8_tensor_to_uint16_range() {
    let input = Tensor::new_with_dtype(vec![0.0, 255.0], vec![1, 2], NumericDType::U8).unwrap();
    let Value::Tensor(out) = call(Value::Tensor(input)) else {
        panic!("expected tensor");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::U16);
    assert_eq!(values(&out), vec![0.0, 65535.0]);
}

#[test]
fn im2uint16_reads_typed_integer_tensor_storage_exactly() {
    let input = Tensor::new_integer(IntegerStorage::U8(vec![0, 255]), vec![1, 2]).unwrap();

    let Value::Tensor(out) = call(Value::Tensor(input)) else {
        panic!("expected tensor");
    };

    assert_eq!(out.numeric_dtype(), NumericDType::U16);
    assert_eq!(
        out.integer_storage(),
        Some(&IntegerStorage::U16(vec![0, u16::MAX]))
    );
}

#[test]
fn converts_int16_tensor_to_uint16_range_with_exact_output_storage() {
    let input =
        Tensor::new_integer(IntegerStorage::I16(vec![i16::MIN, i16::MAX]), vec![1, 2]).unwrap();
    let Value::Tensor(out) = call(Value::Tensor(input)) else {
        panic!("expected tensor");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::U16);
    assert_eq!(
        out.integer_storage(),
        Some(&IntegerStorage::U16(vec![0, u16::MAX]))
    );
}

#[test]
fn clamps_and_rounds_float_tensor_to_uint16() {
    let input = Tensor::new(vec![-0.1, 0.0, 0.5, 1.0, 1.2, f64::NAN], vec![2, 3]).unwrap();
    let Value::Tensor(out) = call(Value::Tensor(input)) else {
        panic!("expected tensor");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::U16);
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(values(&out), vec![0.0, 0.0, 32768.0, 65535.0, 65535.0, 0.0]);
}

#[test]
fn converts_logical_array_to_uint16_extrema() {
    let logical = LogicalArray::new(vec![0, 1, 1, 0], vec![2, 2]).unwrap();
    let Value::Tensor(out) = call(Value::LogicalArray(logical)) else {
        panic!("expected tensor");
    };
    assert_eq!(out.numeric_dtype(), NumericDType::U16);
    assert_eq!(values(&out), vec![0.0, 65535.0, 65535.0, 0.0]);
}

#[test]
fn scales_int16_scalar_and_rejects_unsupported_integer_classes() {
    assert_eq!(
        call(Value::Int(IntValue::I16(i16::MIN))),
        Value::Int(IntValue::U16(0))
    );
    assert_eq!(
        call(Value::Int(IntValue::I16(i16::MAX))),
        Value::Int(IntValue::U16(u16::MAX))
    );
    for (scalar, storage) in [
        (IntValue::I8(0), IntegerStorage::I8(vec![0])),
        (IntValue::I32(0), IntegerStorage::I32(vec![0])),
        (IntValue::I64(0), IntegerStorage::I64(vec![0])),
        (IntValue::U32(0), IntegerStorage::U32(vec![0])),
        (IntValue::U64(0), IntegerStorage::U64(vec![0])),
    ] {
        let scalar_err = block_on(im2uint16_builtin(Value::Int(scalar), Vec::new())).unwrap_err();
        assert_eq!(
            scalar_err.identifier(),
            IM2UINT16_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
        );
        let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        let tensor_err =
            block_on(im2uint16_builtin(Value::Tensor(tensor), Vec::new())).unwrap_err();
        assert_eq!(
            tensor_err.identifier(),
            IM2UINT16_ERROR_UNSUPPORTED_INPUT_TYPE.identifier
        );
    }
}

#[test]
fn im2uint16_descriptor_signatures_cover_surface() {
    let labels: Vec<&str> = IM2UINT16_DESCRIPTOR
        .signatures
        .iter()
        .map(|signature| signature.label)
        .collect();
    assert_eq!(
        labels,
        vec!["J = im2uint16(I)", "J = im2uint16(I, \"indexed\")"]
    );
}

#[test]
fn indexed_mode_offsets_double_and_preserves_integer_indices() {
    let Value::Tensor(double_out) = block_on(im2uint16_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 65536.0], vec![1, 3]).unwrap()),
        vec![Value::String("indexed".into())],
    ))
    .expect("indexed double") else {
        panic!("expected tensor");
    };
    assert_eq!(
        double_out.integer_storage(),
        Some(&IntegerStorage::U16(vec![0, 1, 65535]))
    );

    let Value::Tensor(uint8_out) = block_on(im2uint16_builtin(
        Value::Tensor(
            Tensor::new_integer(IntegerStorage::U8(vec![0, 1, 255]), vec![1, 3]).unwrap(),
        ),
        vec![Value::String("indexed".into())],
    ))
    .expect("indexed uint8") else {
        panic!("expected tensor");
    };
    assert_eq!(
        uint8_out.integer_storage(),
        Some(&IntegerStorage::U16(vec![0, 1, 255]))
    );
}

#[test]
fn indexed_mode_maps_logical_to_zero_and_one_and_checks_double_range() {
    let Value::Tensor(logical_out) = block_on(im2uint16_builtin(
        Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
        vec![Value::String("indexed".into())],
    ))
    .expect("indexed logical") else {
        panic!("expected tensor");
    };
    assert_eq!(
        logical_out.integer_storage(),
        Some(&IntegerStorage::U16(vec![0, 1]))
    );

    let err = block_on(im2uint16_builtin(
        Value::Tensor(Tensor::new(vec![65537.0], vec![1, 1]).unwrap()),
        vec![Value::String("indexed".into())],
    ))
    .expect_err("out-of-range indexed image");
    assert_eq!(err.identifier(), IM2UINT16_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn im2uint16_descriptor_errors_have_stable_codes() {
    let codes: Vec<&str> = IM2UINT16_DESCRIPTOR
        .errors
        .iter()
        .map(|error| error.code)
        .collect();
    assert!(codes.contains(&"RM.IM2UINT16.TOO_MANY_INPUTS"));
    assert!(codes.contains(&"RM.IM2UINT16.TOO_MANY_OUTPUTS"));
    assert!(codes.contains(&"RM.IM2UINT16.INVALID_INPUT"));
    assert!(codes.contains(&"RM.IM2UINT16.UNSUPPORTED_INPUT_TYPE"));
    assert!(codes.contains(&"RM.IM2UINT16.INTERNAL"));
}

#[test]
fn im2uint16_too_many_args_uses_stable_identifier() {
    let err = block_on(im2uint16_builtin(
        Value::Num(1.0),
        vec![Value::from("indexed"), Value::Num(2.0)],
    ))
    .expect_err("expected argument error");
    assert_eq!(err.identifier(), IM2UINT16_ERROR_TOO_MANY_INPUTS.identifier);
}

#[test]
fn im2uint16_resident_uint8_input_restores_exact_integer_output_to_owner() {
    test_support::with_test_provider(|provider| {
        let input = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&[0, 128, 255]),
                shape: &[1, 3],
            })
            .expect("upload uint8 image");
        let Value::GpuTensor(output) =
            block_on(im2uint16_builtin(Value::GpuTensor(input), Vec::new()))
                .expect("resident im2uint16")
        else {
            panic!("expected resident uint16 output");
        };
        assert_eq!(output.device_id(), provider.device_id());
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(IntegerElementType::U16)
        );
        assert_eq!(
            block_on(provider.download_integer(&output))
                .expect("download uint16 image")
                .data,
            HostIntegerDataOwned::U16(vec![0, 32_896, 65_535])
        );
    });
}

#[test]
fn empty_nd_image_preserves_geometry_and_output_class() {
    let image = Tensor::new_integer(IntegerStorage::U8(vec![]), vec![0, 2, 3]).unwrap();
    let Value::Tensor(output) = call(Value::Tensor(image)) else {
        panic!("empty image must remain a tensor");
    };
    assert_eq!(output.shape(), &[0, 2, 3]);
    assert_eq!(output.numeric_dtype(), NumericDType::U16);
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
        let err = block_on(im2uint16_builtin(
            Value::GpuTensor(input),
            vec![Value::String("not-indexed".into())],
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), IM2UINT16_ERROR_INVALID_INPUT.identifier);
    });
}
