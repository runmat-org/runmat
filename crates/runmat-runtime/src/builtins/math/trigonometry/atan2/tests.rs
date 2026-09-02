use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{CharArray, IntegerStorage, LogicalArray, Tensor, Value};
use std::f64::consts::PI;

const EPS: f64 = 1e-12;

fn atan2_builtin(y: Value, x: Value) -> BuiltinResult<Value> {
    block_on(super::atan2_builtin(y, x))
}

fn error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_scalar_pair() {
    let result = atan2_builtin(Value::Num(1.0), Value::Num(1.0)).expect("atan2");
    match result {
        Value::Num(v) => assert!((v - PI / 4.0).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_quadrant_detection() {
    let result = atan2_builtin(Value::Num(-1.0), Value::Num(-1.0)).expect("atan2");
    match result {
        Value::Num(v) => assert!((v + 3.0 * PI / 4.0).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_matrix_vs_scalar_broadcast() {
    let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let result = atan2_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("broadcast");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [
                (1.0f64).atan2(2.0),
                (2.0f64).atan2(2.0),
                (3.0f64).atan2(2.0),
                (4.0f64).atan2(2.0),
            ];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS, "{actual} vs {expect}");
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_row_vector_broadcast() {
    let y = Tensor::new(vec![1.0, -1.0, 2.0, -2.0], vec![2, 2]).unwrap();
    let x = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
    let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("row broadcast");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [
                (1.0f64).atan2(1.0),
                (-1.0f64).atan2(1.0),
                (2.0f64).atan2(1.0),
                (-2.0f64).atan2(1.0),
            ];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn atan2_maps_tabular_variables_and_preserves_timetable_metadata() {
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
        ],
    )
    .unwrap();
    let Value::Object(mut timetable) = table else {
        panic!("expected table object")
    };
    timetable.class_name = runmat_types::standard::TIMETABLE.owned();
    timetable
        .properties
        .insert("RowTimes".into(), Value::from("preserved"));

    let Value::Object(output) = atan2_builtin(Value::Object(timetable), Value::Num(-1.0)).unwrap()
    else {
        panic!("expected timetable output")
    };
    assert!(output.is_class(runmat_types::standard::TIMETABLE));
    assert_eq!(
        output.properties.get("RowTimes"),
        Some(&Value::from("preserved"))
    );
    let variables = crate::builtins::table::table_variables(&output).unwrap();
    for (actual, expected) in tensor::value_into_tensor_for("test", variables.fields["A"].clone())
        .unwrap()
        .materialize_f64()
        .iter()
        .zip([3.0 * PI / 4.0, -3.0 * PI / 4.0])
    {
        assert!((actual - expected).abs() < EPS);
    }
    for (actual, expected) in tensor::value_into_tensor_for("test", variables.fields["B"].clone())
        .unwrap()
        .materialize_f64()
        .iter()
        .zip([PI, 3.0 * PI / 4.0])
    {
        assert!((actual - expected).abs() < EPS);
    }
}

#[test]
fn atan2_rejects_mismatched_tabular_variables() {
    let table = |name: &str| {
        crate::builtins::table::table_from_columns(
            vec![name.into()],
            vec![Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap())],
        )
        .unwrap()
    };
    let error = atan2_builtin(table("A"), table("B"))
        .expect_err("mismatched tabular variables must reject");
    assert_eq!(error.identifier(), ATAN2_ERROR_INVALID_INPUT.identifier);
    assert!(error
        .message()
        .contains("matching variable names and order"));
}

#[test]
fn atan2_typed_integer_tensors_read_exact_storage() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = Tensor::new_integer(IntegerStorage::I16(vec![1, -1, 2, -2]), vec![2, 2]).unwrap();
    let x = Tensor::new_integer(IntegerStorage::I16(vec![1, 1]), vec![1, 2]).unwrap();

    let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [
                (1.0f64).atan2(1.0),
                (-1.0f64).atan2(1.0),
                (2.0f64).atan2(1.0),
                (-2.0f64).atan2(1.0),
            ];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn atan2_scalar_fast_path_reads_typed_integer_without_double_mirror() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let y = Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).unwrap();
    let x = Tensor::new_integer(IntegerStorage::I16(vec![1]), vec![1, 1]).unwrap();

    let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
    match result {
        Value::Num(v) => assert!((v - 1.0f64.atan2(1.0)).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_char_input() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let chars = CharArray::new("A".chars().collect(), 1, 1).unwrap();
    let result = atan2_builtin(Value::CharArray(chars), Value::Num(100.0)).expect("atan2");
    match result {
        Value::Num(v) => assert!((v - (65.0f64).atan2(100.0)).abs() < EPS),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_logical_input() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let logical = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
    let x = Tensor::new(vec![1.0, 1.0, -1.0, -1.0], vec![2, 2]).unwrap();
    let result =
        atan2_builtin(Value::LogicalArray(logical), Value::Tensor(x)).expect("logical atan2");
    match result {
        Value::Tensor(t) => {
            let expected = [
                1.0f64.atan2(1.0),
                0.0f64.atan2(1.0),
                0.0f64.atan2(-1.0),
                1.0f64.atan2(-1.0),
            ];
            for (actual, expect) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((actual - expect).abs() < EPS);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_zero_zero_is_zero() {
    let result = atan2_builtin(Value::Num(0.0), Value::Num(0.0)).expect("atan2");
    match result {
        Value::Num(v) => assert_eq!(v, 0.0),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_signed_zero_behaviour() {
    let neg_zero = f64::from_bits(0x8000_0000_0000_0000);
    let Value::Num(positive_zero_case) =
        atan2_builtin(Value::Num(0.0), Value::Num(neg_zero)).expect("atan2")
    else {
        panic!("expected numeric result");
    };
    assert_eq!(positive_zero_case.to_bits(), 0.0f64.to_bits());

    let Value::Num(negative_zero_pair) =
        atan2_builtin(Value::Num(neg_zero), Value::Num(neg_zero)).expect("atan2")
    else {
        panic!("expected numeric result");
    };
    assert_eq!(negative_zero_pair.to_bits(), 0.0f64.to_bits());

    let Value::Num(neg_zero_result) =
        atan2_builtin(Value::Num(neg_zero), Value::Num(0.0)).expect("atan2")
    else {
        panic!("expected numeric result");
    };
    assert_eq!(
        neg_zero_result.to_bits(),
        f64::from_bits(0x8000_0000_0000_0000).to_bits(),
        "expected negative zero, got {neg_zero_result}"
    );
}

#[test]
fn atan2_mixed_single_double_computes_in_double_and_returns_single() {
    let y =
        Tensor::from_numeric_storage(NumericStorage::F32(vec![1.4e32_f32]), vec![1, 1]).unwrap();
    let x = Tensor::new(vec![-5.305e32], vec![1, 1]).unwrap();
    let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
    let Value::Tensor(tensor) = result else {
        panic!("expected native-single scalar tensor");
    };
    assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
    let expected = matlab_atan2_f64(1.4e32_f32 as f64, -5.305e32) as f32;
    assert_eq!(tensor.as_f32_slice(), Some([expected].as_slice()));
}

#[test]
fn atan2_nonfloating_inputs_are_independently_gated() {
    let integer = atan2_builtin(
        Value::Int(runmat_value::IntValue::U64(u64::MAX)),
        Value::Num(1.0),
    )
    .expect_err("integer input must be gated");
    assert_eq!(
        integer.identifier(),
        ATAN2_INTEGER_INPUT_EXTENSION.error_identifier
    );
    let logical =
        atan2_builtin(Value::Bool(true), Value::Num(1.0)).expect_err("logical input must be gated");
    assert_eq!(
        logical.identifier(),
        ATAN2_LOGICAL_INPUT_EXTENSION.error_identifier
    );
    let chars = CharArray::new_row("A");
    let character = atan2_builtin(Value::CharArray(chars), Value::Num(1.0))
        .expect_err("character input must be gated");
    assert_eq!(
        character.identifier(),
        ATAN2_CHARACTER_INPUT_EXTENSION.error_identifier
    );
}

#[test]
fn atan2_integer_extension_covers_all_eight_classes_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let values = [
        runmat_value::IntValue::I8(i8::MAX),
        runmat_value::IntValue::I16(i16::MAX),
        runmat_value::IntValue::I32(i32::MAX),
        runmat_value::IntValue::I64(i64::MAX),
        runmat_value::IntValue::U8(u8::MAX),
        runmat_value::IntValue::U16(u16::MAX),
        runmat_value::IntValue::U32(u32::MAX),
        runmat_value::IntValue::U64(u64::MAX),
    ];
    for value in values {
        let expected = matlab_atan2_f64(value.to_f64(), 1.0);
        let result = atan2_builtin(Value::Int(value), Value::Num(1.0)).expect("atan2");
        let Value::Num(actual) = result else {
            panic!("expected double scalar result");
        };
        assert_eq!(actual, expected);
    }
}

#[test]
fn atan2_rejects_excess_outputs() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = atan2_builtin(Value::Num(1.0), Value::Num(1.0))
        .expect_err("second output must be rejected");
    assert_eq!(error.identifier(), ATAN2_ERROR_TOO_MANY_OUTPUTS.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_empty_tensor_result() {
    let y = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let x = Tensor::new(Vec::new(), vec![0, 3]).unwrap();
    let result = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).expect("atan2");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![0, 3]);
            assert!(t.materialize_f64().is_empty());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_complex_input_errors() {
    let err = atan2_builtin(Value::Complex(1.0, 1.0), Value::Num(1.0)).unwrap_err();
    assert_eq!(err.identifier(), ATAN2_ERROR_COMPLEX_UNSUPPORTED.identifier);
    let message = error_message(err);
    assert!(message.to_ascii_lowercase().contains("complex"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_dimension_mismatch_errors() {
    let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
    let x = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
    let err = atan2_builtin(Value::Tensor(y), Value::Tensor(x)).unwrap_err();
    assert_eq!(err.identifier(), ATAN2_ERROR_SIZE_MISMATCH.identifier);
    let message = error_message(err);
    assert!(
        message.to_ascii_lowercase().contains("size"),
        "unexpected error: {message}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let neg_zero = f64::from_bits(0x8000_0000_0000_0000);
        let y = Tensor::new(
            vec![1.0, 1.0, -1.0, -1.0, 0.0, neg_zero, neg_zero],
            vec![1, 7],
        )
        .unwrap();
        let x = Tensor::new(
            vec![1.0, -1.0, 1.0, -1.0, neg_zero, neg_zero, 0.0],
            vec![1, 7],
        )
        .unwrap();
        let hy = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &y.materialize_f64(),
                shape: &y.shape,
            })
            .expect("upload y");
        let hx = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.materialize_f64(),
                shape: &x.shape,
            })
            .expect("upload x");
        let result = atan2_builtin(Value::GpuTensor(hy), Value::GpuTensor(hx)).expect("gpu atan2");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 7]);
        let expected = [
            (1.0f64).atan2(1.0),
            (1.0f64).atan2(-1.0),
            (-1.0f64).atan2(1.0),
            (-1.0f64).atan2(-1.0),
            0.0,
            0.0,
            neg_zero,
        ];
        for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((actual - expect).abs() < EPS);
        }
        let values = gathered.materialize_f64();
        assert_eq!(values[4].to_bits(), 0.0f64.to_bits());
        assert_eq!(values[5].to_bits(), 0.0f64.to_bits());
        assert_eq!(values[6].to_bits(), neg_zero.to_bits());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn atan2_gpu_host_mix_falls_back() {
    test_support::with_test_provider(|provider| {
        let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let hy = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &y.materialize_f64(),
                shape: &y.shape,
            })
            .expect("upload y");
        let result = atan2_builtin(Value::GpuTensor(hy), Value::Num(2.0)).expect("atan2");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        let expected = [(1.0f64).atan2(2.0), (2.0f64).atan2(2.0)];
        for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((actual - expect).abs() < EPS);
        }
    });
}

#[test]
fn atan2_gpu_host_mix_reads_typed_integer_rhs_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let y = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let hy = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &y.materialize_f64(),
                shape: &y.shape,
            })
            .expect("upload y");
        let x = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap();

        let result = atan2_builtin(Value::GpuTensor(hy), Value::Tensor(x)).expect("atan2");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert!((gathered.materialize_f64()[0] - 1.0f64.atan2(1.0)).abs() < EPS);
        assert!((gathered.materialize_f64()[1] - 2.0f64.atan2(2.0)).abs() < EPS);
    });
}

#[test]
fn atan2_gpu_host_mix_reads_typed_integer_lhs_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let x = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let hx = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &x.materialize_f64(),
                shape: &x.shape,
            })
            .expect("upload x");
        let y = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1]).unwrap();

        let result = atan2_builtin(Value::Tensor(y), Value::GpuTensor(hx)).expect("atan2");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert!((gathered.materialize_f64()[0] - 1.0f64.atan2(1.0)).abs() < EPS);
        assert!((gathered.materialize_f64()[1] - 2.0f64.atan2(2.0)).abs() < EPS);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn atan2_wgpu_matches_cpu_elementwise() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let neg_zero = f64::from_bits(0x8000_0000_0000_0000);
    let y = Tensor::new(vec![0.0, neg_zero, neg_zero, 1.0, -1.0, 2.0], vec![2, 3]).unwrap();
    let x = Tensor::new(vec![neg_zero, neg_zero, 0.0, 1.0, 1.0, -1.0], vec![2, 3]).unwrap();
    let cpu = atan2_host(Value::Tensor(y.clone()), Value::Tensor(x.clone())).unwrap();
    let hy = runmat_accelerate_api::provider()
        .unwrap()
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &y.materialize_f64(),
            shape: &y.shape,
        })
        .unwrap();
    let hx = runmat_accelerate_api::provider()
        .unwrap()
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &x.materialize_f64(),
            shape: &x.shape,
        })
        .unwrap();
    let gpu = block_on(atan2_gpu_pair(hy, hx)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(ct) => {
            assert_eq!(ct.shape, gathered.shape);
            let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
            };
            for (actual, expect) in gathered
                .materialize_f64()
                .iter()
                .zip(ct.materialize_f64().iter())
            {
                assert!((actual - expect).abs() < tol, "{actual} vs {expect}");
            }
            let values = gathered.materialize_f64();
            assert_eq!(values[0].to_bits(), 0.0f64.to_bits());
            assert_eq!(values[1].to_bits(), 0.0f64.to_bits());
            assert_eq!(values[2].to_bits(), neg_zero.to_bits());
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}
