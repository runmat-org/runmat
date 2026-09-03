use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use crate::{BuiltinResult, RuntimeError};
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;

fn assert_error_contains(err: &RuntimeError, expected: &str) {
    assert!(
        err.message().contains(expected),
        "unexpected error: {}",
        err.message()
    );
}

fn run_xor(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(super::xor_builtin(lhs, rhs))
}

#[cfg(feature = "wgpu")]
fn run_xor_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(xor_host(lhs, rhs))
}
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::{CharArray, IntValue, IntegerStorage, LogicalArray, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_of_booleans() {
    assert_eq!(
        run_xor(Value::Bool(true), Value::Bool(false)).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run_xor(Value::Bool(true), Value::Bool(true)).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_xor(Value::Bool(false), Value::Bool(false)).unwrap(),
        Value::Bool(false)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_numeric_arrays() {
    let a = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![3.0, 4.0, 0.0, 0.0], vec![2, 2]).unwrap();
    let result = run_xor(Value::Tensor(a), Value::Tensor(b)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 2]);
            assert_eq!(array.data, vec![0, 1, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_logical_array_inputs() {
    let left = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
    let right = LogicalArray::new(vec![0, 1, 1], vec![3, 1]).unwrap();
    let result = run_xor(Value::LogicalArray(left), Value::LogicalArray(right)).expect("xor");
    match result {
        Value::LogicalArray(arr) => {
            assert_eq!(arr.shape, vec![3, 1]);
            assert_eq!(arr.data, vec![1, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_empty_inputs() {
    let lhs = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
    let rhs = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
    let result = run_xor(Value::Tensor(lhs), Value::Tensor(rhs)).expect("xor");
    match result {
        Value::LogicalArray(arr) => {
            assert_eq!(arr.shape, vec![0, 3]);
            assert!(arr.data.is_empty());
        }
        other => panic!("expected empty logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_scalar_broadcasts() {
    let tensor = Tensor::new(vec![1.0, 0.0, 3.0, 0.0], vec![4, 1]).unwrap();
    let result = run_xor(Value::Tensor(tensor), Value::Int(IntValue::I32(1))).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![4, 1]);
            assert_eq!(array.data, vec![0, 1, 0, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_reads_typed_integer_storage_exactly_for_truth_values() {
    let lhs = Tensor::new_integer(
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
        vec![1, 3],
    )
    .unwrap();
    let rhs = Tensor::new_integer(IntegerStorage::I16(vec![0, -2, -3]), vec![1, 3]).unwrap();
    let result = run_xor(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(array.data, vec![0, 0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }

    assert_eq!(
        run_xor(Value::Int(IntValue::U64(u64::MAX)), Value::Bool(false)).unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn xor_accepts_every_integer_class_without_floating_truth_conversion() {
    let cases = [
        IntegerStorage::I8(vec![0, i8::MIN]),
        IntegerStorage::I16(vec![0, i16::MIN]),
        IntegerStorage::I32(vec![0, i32::MIN]),
        IntegerStorage::I64(vec![0, i64::MIN]),
        IntegerStorage::U8(vec![0, u8::MAX]),
        IntegerStorage::U16(vec![0, u16::MAX]),
        IntegerStorage::U32(vec![0, u32::MAX]),
        IntegerStorage::U64(vec![0, u64::MAX]),
    ];
    for storage in cases {
        let input = Tensor::new_integer(storage, vec![1, 2]).expect("integer input");
        let result = run_xor(Value::Tensor(input), Value::Bool(true)).expect("xor");
        let Value::LogicalArray(output) = result else {
            panic!("expected logical array");
        };
        assert_eq!(output.data, vec![1, 0]);
    }
}

#[test]
fn xor_explicit_resident_integer_fallback_is_exact_and_stays_resident() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![0, (1_u64 << 53) + 1, u64::MAX]),
            vec![1, 3],
        )
        .expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let result = run_xor(Value::GpuTensor(handle), Value::Bool(true)).expect("xor");
        let Value::GpuTensor(output) = &result else {
            panic!("explicit gpuArray result must remain resident");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output));
        assert!(runmat_accelerate_api::handle_is_logical(output));
        assert_eq!(
            test_support::gather(result)
                .expect("gather")
                .materialize_f64(),
            vec![1.0, 0.0, 0.0]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_char_arrays_are_available_in_matlab_compatibility_mode() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(false);
    let lhs = CharArray::new(vec!['R', 'u', '\0'], 1, 3).unwrap();
    let rhs = CharArray::new(vec!['R', '\0', 'n'], 1, 3).unwrap();
    let result = run_xor(Value::CharArray(lhs), Value::CharArray(rhs)).expect("xor char arrays");
    match result {
        Value::LogicalArray(arr) => {
            assert_eq!(arr.shape, vec![1, 3]);
            assert_eq!(arr.data, vec![0, 1, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_treats_nan_as_true() {
    let result = run_xor(Value::Num(f64::NAN), Value::Num(1.0)).unwrap();
    assert_eq!(result, Value::Bool(false));
    let result = run_xor(Value::Num(f64::NAN), Value::Num(0.0)).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_complex_inputs() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = run_xor(Value::Complex(0.0, 0.0), Value::Complex(0.0, 2.0)).unwrap();
    assert_eq!(result, Value::Bool(true));

    let result = run_xor(Value::Complex(1.0, 0.0), Value::Complex(0.0, 2.0)).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_size_mismatch_errors() {
    let lhs = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
    let err = run_xor(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap_err();
    assert_error_contains(&err, "size mismatch");
    assert_eq!(err.identifier(), XOR_ERROR_SIZE_MISMATCH.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_rejects_unsupported_types() {
    let err = run_xor(Value::String("runmat".into()), Value::Bool(true)).unwrap_err();
    assert_error_contains(&err, "unsupported input type");
    assert_eq!(err.identifier(), XOR_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![2, 2]).unwrap();
        let other = Tensor::new(vec![1.0, 0.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view_a = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let view_b = HostTensorView {
            data: &other.materialize_f64(),
            shape: &other.shape,
        };
        let a = provider.upload(&view_a).unwrap();
        let b = provider.upload(&view_b).unwrap();
        let result = run_xor(Value::GpuTensor(a), Value::GpuTensor(b)).unwrap();
        let gathered = test_support::gather(result).unwrap();
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0, 1.0, 0.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xor_gpu_supports_broadcast() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![4, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();

        let view_lhs = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        };
        let view_rhs = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        };

        let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
        let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");

        let result = run_xor(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs)).expect("xor");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![4, 1]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn xor_wgpu_integer_handle_uses_exact_fallback_and_preserves_explicit_residency() {
    let _accel_guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let input = Tensor::new_integer(
        IntegerStorage::U64(vec![0, (1_u64 << 53) + 1, u64::MAX]),
        vec![1, 3],
    )
    .expect("integer input");
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
    let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
    let result = run_xor(Value::GpuTensor(handle), Value::Bool(true)).expect("xor");
    let Value::GpuTensor(output) = &result else {
        panic!("explicit gpuArray result must remain resident");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(output));
    assert_eq!(
        test_support::gather(result)
            .expect("gather")
            .materialize_f64(),
        vec![1.0, 0.0, 0.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn xor_wgpu_matches_host_path() {
    let _accel_guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };

    let lhs = Tensor::new(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![1.0, 0.0, 3.0, 4.0], vec![2, 2]).unwrap();

    let cpu_value =
        run_xor_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).expect("host xor");
    let (expected_data, expected_shape) = match cpu_value {
        Value::LogicalArray(arr) => (arr.data.clone(), arr.shape.clone()),
        other => panic!("expected logical array, got {other:?}"),
    };

    let view_lhs = HostTensorView {
        data: &lhs.materialize_f64(),
        shape: &lhs.shape,
    };
    let view_rhs = HostTensorView {
        data: &rhs.materialize_f64(),
        shape: &rhs.shape,
    };
    let gpu_lhs = provider.upload(&view_lhs).expect("upload lhs");
    let gpu_rhs = provider.upload(&view_rhs).expect("upload rhs");

    let gpu_value = run_xor(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs)).expect("gpu xor");
    let gathered = test_support::gather(gpu_value).expect("gather gpu result");

    assert_eq!(gathered.shape, expected_shape);
    let tol = match provider.precision() {
        ProviderPrecision::F64 => 1e-12,
        ProviderPrecision::F32 => 1e-5,
    };
    for (idx, (actual, expected)) in gathered
        .materialize_f64()
        .iter()
        .zip(expected_data.iter())
        .enumerate()
    {
        let expected_f = if *expected != 0 { 1.0 } else { 0.0 };
        assert!(
            (actual - expected_f).abs() <= tol,
            "mismatch at index {idx}: got {actual}, expected {expected_f}"
        );
    }
}
