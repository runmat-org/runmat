use super::*;
use crate::builtins::common::test_support;
use crate::RuntimeError;
use futures::executor::block_on;
use runmat_value::{CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, Tensor};

fn mod_builtin(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(super::mod_builtin(lhs, rhs))
}

#[test]
fn mod_real_arrays_preserve_native_single_storage_including_empty() {
    let lhs = Tensor::from_f32(vec![5.5, -5.5], vec![1, 2]).unwrap();
    let rhs = Tensor::from_f32(vec![2.0, 2.0], vec![1, 2]).unwrap();
    let output = compute_mod_real(&lhs, &rhs).unwrap();
    let Value::Tensor(output) = output else {
        panic!("expected native-single tensor")
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        runmat_value::NumericStorage::F32(vec![1.5, 0.5])
    );

    let lhs = Tensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let rhs = Tensor::from_f32(Vec::new(), vec![0, 2]).unwrap();
    let Value::Tensor(output) = compute_mod_real(&lhs, &rhs).unwrap() else {
        panic!("expected empty native-single tensor")
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        runmat_value::NumericStorage::F32(Vec::new())
    );
}

#[test]
fn mod_mixed_single_and_double_computes_in_single_precision() {
    let lhs = Tensor::from_f32(vec![-3.5, 4.5], vec![1, 2]).unwrap();
    let rhs = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
    let Value::Tensor(output) = compute_mod_real(&lhs, &rhs).unwrap() else {
        panic!("expected native-single tensor")
    };
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        runmat_value::NumericStorage::F32(vec![0.5, 0.5])
    );
}

#[test]
fn mod_maps_table_variables_and_preserves_container() {
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            Value::Tensor(Tensor::new(vec![5.0, 8.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![7.0, 11.0], vec![2, 1]).unwrap()),
        ],
    )
    .unwrap();
    let Value::Object(output) = mod_builtin(table, Value::Num(3.0)).unwrap() else {
        panic!("expected table output")
    };
    let variables = crate::builtins::table::table_variables(&output).unwrap();
    assert_eq!(
        variables.fields["A"].clone(),
        Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![2, 1]).unwrap())
    );
    assert_eq!(
        variables.fields["B"].clone(),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap())
    );
}

#[test]
fn mod_duration_returns_duration_in_source_format() {
    let duration = crate::builtins::duration::duration_object_from_days_tensor(
        Tensor::new(vec![25.0 / 24.0, 50.0 / 24.0], vec![1, 2]).unwrap(),
        "hh:mm:ss",
    )
    .unwrap();
    let divisor = crate::builtins::duration::duration_object_from_days_tensor(
        Tensor::new(vec![1.0], vec![1, 1]).unwrap(),
        "hh:mm:ss",
    )
    .unwrap();
    let result = mod_builtin(duration, divisor).unwrap();
    assert!(crate::builtins::duration::is_duration_object(&result));
    let days = crate::builtins::duration::duration_tensor_from_duration_value(&result).unwrap();
    let expected = [1.0 / 24.0, 2.0 / 24.0];
    for (actual, expected) in days.materialize_f64().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn mod_rejects_excess_outputs() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error =
        mod_builtin(Value::Num(5.0), Value::Num(3.0)).expect_err("mod must reject excess outputs");
    assert_eq!(error.identifier(), Some("RunMat:mod:TooManyOutputs"));
}

fn assert_error_contains(error: RuntimeError, needle: &str) {
    assert!(
        error.message().contains(needle),
        "unexpected error: {}",
        error.message()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_positive_values() {
    let result = mod_builtin(Value::Num(17.0), Value::Num(5.0)).expect("mod");
    match result {
        Value::Num(v) => assert!((v - 2.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_negative_divisor_keeps_sign() {
    let tensor = Tensor::new(vec![-7.0, -3.0, 4.0, 9.0], vec![4, 1]).unwrap();
    let divisor = Tensor::new(vec![-4.0], vec![1, 1]).unwrap();
    let result = mod_builtin(Value::Tensor(tensor), Value::Tensor(divisor)).expect("mod broadcast");
    match result {
        Value::Tensor(out) => {
            assert_eq!(out.materialize_f64(), vec![-3.0, -3.0, 0.0, -3.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_negative_numerator_positive_divisor() {
    let result = mod_builtin(Value::Num(-3.0), Value::Num(2.0)).expect("mod");
    match result {
        Value::Num(v) => assert!((v - 1.0).abs() < 1e-12),
        other => panic!("expected scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_zero_divisor_returns_dividend() {
    let result = mod_builtin(Value::Num(3.0), Value::Num(0.0)).expect("mod");
    match result {
        Value::Num(v) => assert_eq!(v, 3.0),
        other => panic!("expected dividend, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_matrix_scalar_broadcast() {
    let matrix = Tensor::new(vec![4.5, 7.1, -2.3, 0.4], vec![2, 2]).unwrap();
    let result = mod_builtin(Value::Tensor(matrix), Value::Num(2.0)).expect("broadcast");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 2]);
            let expected = [0.5, 1.1, 1.7, 0.4];
            for (a, b) in t.materialize_f64().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_rejects_complex_operands() {
    let complex =
        ComplexTensor::new(vec![(3.0, 4.0), (-2.0, 5.0)], vec![1, 2]).expect("complex tensor");
    let divisor = ComplexTensor::new(vec![(2.0, 1.0)], vec![1, 1]).expect("divisor");
    assert!(mod_builtin(Value::ComplexTensor(complex), Value::ComplexTensor(divisor)).is_err());
    assert!(mod_builtin(Value::Complex(2.0, 0.0), Value::Num(1.0)).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_char_array_support() {
    let chars = CharArray::new("ABC".chars().collect(), 1, 3).unwrap();
    let result = mod_builtin(Value::CharArray(chars), Value::Num(5.0)).expect("mod");
    match result {
        Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![0.0, 1.0, 2.0]),
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_string_input_errors() {
    let err =
        mod_builtin(Value::from("abc"), Value::Num(3.0)).expect_err("string inputs should error");
    let identifier = err.identifier().map(str::to_string);
    assert_error_contains(err, "expected numeric input");
    assert_eq!(identifier.as_deref(), MOD_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_logical_array_support() {
    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
    let value = mod_builtin(Value::LogicalArray(logical), Value::Num(2.0)).expect("logical mod");
    match value {
        Value::Tensor(t) => assert_eq!(t.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0]),
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_vector_broadcasting() {
    let lhs = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
    let rhs = Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).unwrap();
    let result = mod_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("vector broadcast");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 3]);
            assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_nan_inputs_propagate() {
    let result = mod_builtin(Value::Num(f64::NAN), Value::Num(3.0)).expect("mod");
    match result {
        Value::Num(v) => assert!(v.is_nan()),
        other => panic!("expected NaN result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_gpu_pair_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![-5.0, -3.0, 0.0, 1.0, 6.0, 9.0], vec![3, 2]).unwrap();
        let divisor = Tensor::new(vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0], vec![3, 2]).unwrap();
        let a_view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let b_view = runmat_accelerate_api::HostTensorView {
            data: &divisor.materialize_f64(),
            shape: &divisor.shape,
        };
        let a_handle = provider.upload(&a_view).expect("upload a");
        let b_handle = provider.upload(&b_view).expect("upload b");
        let result =
            mod_builtin(Value::GpuTensor(a_handle), Value::GpuTensor(b_handle)).expect("mod");
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.shape, vec![3, 2]);
        assert_eq!(
            gathered.materialize_f64(),
            vec![3.0, 1.0, 0.0, 1.0, 2.0, 1.0]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_int_scalar_preserves_exact_class() {
    let result =
        mod_builtin(Value::Int(IntValue::I32(-7)), Value::Int(IntValue::I32(4))).expect("mod");
    match result {
        Value::Int(IntValue::I32(v)) => assert_eq!(v, 1),
        other => panic!("expected int32 scalar result, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mod_scalar_fast_path_reads_typed_integer_storage_exactly() {
    let lhs = Tensor::new_integer(IntegerStorage::I16(vec![7]), vec![1, 1]).expect("lhs tensor");

    assert_eq!(scalar_real_value(&Value::Tensor(lhs.clone())), Some(7.0));

    let result = mod_builtin(Value::Tensor(lhs), Value::Num(4.0)).expect("mod");
    match result {
        Value::Int(IntValue::I16(v)) => assert_eq!(v, 3),
        other => panic!("expected int16 scalar result, got {other:?}"),
    }
}

#[test]
fn mod_dense_integer_arrays_preserve_exact_storage_without_mirror() {
    let lhs = Tensor::new_integer(IntegerStorage::I64(vec![-7, 7]), vec![2, 1]).expect("lhs");
    let rhs = Tensor::new_integer(IntegerStorage::I64(vec![4, -4, 0]), vec![1, 3]).expect("rhs");

    let result = mod_builtin(Value::Tensor(lhs), Value::Tensor(rhs)).expect("mod");
    let Value::Tensor(result) = result else {
        panic!("expected integer tensor");
    };
    assert_eq!(result.shape, vec![2, 3]);
    assert_eq!(
        result.integer_storage(),
        Some(&IntegerStorage::I64(vec![1, 3, -3, -1, -7, 7]))
    );

    let lhs = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("lhs");
    assert_eq!(
        mod_builtin(Value::Tensor(lhs), Value::Num(3.0)).expect("mod"),
        Value::Int(IntValue::U64(0))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn mod_wgpu_matches_cpu() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let numer = Tensor::new(vec![-5.0, -3.25, 0.0, 1.75, 6.5, 9.0], vec![3, 2]).unwrap();
    let denom = Tensor::new(vec![4.0, -2.5, 3.0, 3.0, 2.0, -5.0], vec![3, 2]).unwrap();
    let cpu_value =
        mod_host(Value::Tensor(numer.clone()), Value::Tensor(denom.clone())).expect("cpu mod");

    let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
    let numer_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &numer.materialize_f64(),
            shape: &numer.shape,
        })
        .expect("upload numer");
    let denom_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &denom.materialize_f64(),
            shape: &denom.shape,
        })
        .expect("upload denom");

    let gpu_value = block_on(mod_gpu_pair(numer_handle, denom_handle)).expect("gpu mod");
    let gpu_tensor = test_support::gather(gpu_value).expect("gather gpu result");

    let cpu_tensor = match cpu_value {
        Value::Tensor(t) => t,
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).expect("scalar tensor"),
        other => panic!("unexpected CPU result {other:?}"),
    };

    assert_eq!(gpu_tensor.shape, cpu_tensor.shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (gpu, cpu) in gpu_tensor
        .materialize_f64()
        .iter()
        .zip(cpu_tensor.materialize_f64().iter())
    {
        assert!(
            (gpu - cpu).abs() <= tol,
            "|{gpu} - {cpu}| exceeded tolerance {tol}"
        );
    }

    let numer = provider
        .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
            data: runmat_accelerate_api::HostIntegerDataView::I64(&[-7, 7, -7, 5]),
            shape: &[2, 2],
        })
        .expect("upload integer numerators");
    let denom = provider
        .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
            data: runmat_accelerate_api::HostIntegerDataView::I64(&[4, -4, 0, 0]),
            shape: &[2, 2],
        })
        .expect("upload integer divisors");
    let Value::GpuTensor(output) =
        block_on(mod_gpu_pair(numer.clone(), denom.clone())).expect("resident integer modulus")
    else {
        panic!("integer modulus must remain resident")
    };
    assert_eq!(
        block_on(provider.download_integer(&output))
            .expect("download integer modulus")
            .data,
        runmat_accelerate_api::HostIntegerDataOwned::I64(vec![1, -1, -7, 5])
    );
    for handle in [&numer, &denom, &output] {
        provider.free(handle).expect("free integer modulus handle");
    }
}
