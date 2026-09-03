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

fn run_and(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(super::and_builtin(lhs, rhs))
}

#[cfg(feature = "wgpu")]
fn run_and_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    block_on(and_host(lhs, rhs))
}
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::{CharArray, ComplexTensor, IntValue, IntegerStorage, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_of_booleans() {
    assert_eq!(
        run_and(Value::Bool(true), Value::Bool(false)).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_and(Value::Bool(true), Value::Bool(true)).unwrap(),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_numeric_arrays() {
    let a = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![3.0, 4.0, 0.0, 0.0], vec![2, 2]).unwrap();
    let result = run_and(Value::Tensor(a), Value::Tensor(b)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 2]);
            assert_eq!(array.data, vec![1, 0, 0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_scalar_broadcasts() {
    let tensor = Tensor::new(vec![1.0, 0.0, 3.0, 0.0], vec![4, 1]).unwrap();
    let result = run_and(Value::Tensor(tensor), Value::Int(IntValue::I32(1))).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![4, 1]);
            assert_eq!(array.data, vec![1, 0, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_reads_typed_integer_storage_exactly_for_truth_values() {
    let lhs = Tensor::new_integer(
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
        vec![1, 3],
    )
    .unwrap();
    let rhs = Tensor::new_integer(IntegerStorage::I16(vec![1, -2, 0]), vec![1, 3]).unwrap();
    let result = run_and(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(array.data, vec![0, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }

    assert_eq!(
        run_and(Value::Int(IntValue::U64(u64::MAX)), Value::Bool(true)).unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run_and(Value::Int(IntValue::U64(u64::MAX)), Value::Num(-0.5)).unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn and_aligns_table_variables_by_identity() {
    let left = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
        ],
    )
    .unwrap();
    let right = crate::builtins::table::table_from_columns(
        vec!["B".into(), "A".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![2, 1]).unwrap()),
        ],
    )
    .unwrap();

    let Value::Object(output) = run_and(left, right).expect("table and") else {
        panic!("expected table output")
    };
    let variables = crate::builtins::table::table_variables(&output).unwrap();
    assert_eq!(variables.fields.keys().collect::<Vec<_>>(), vec!["A", "B"]);
    assert!(matches!(
        &variables.fields["A"],
        Value::LogicalArray(values) if values.data == vec![1, 0]
    ));
    assert!(matches!(
        &variables.fields["B"],
        Value::LogicalArray(values) if values.data == vec![0, 1]
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_accepts_all_integer_classes_independently() {
    let storages = [
        IntegerStorage::I8(vec![0, -1]),
        IntegerStorage::I16(vec![0, -2]),
        IntegerStorage::I32(vec![0, -3]),
        IntegerStorage::I64(vec![0, i64::MIN]),
        IntegerStorage::U8(vec![0, 1]),
        IntegerStorage::U16(vec![0, 2]),
        IntegerStorage::U32(vec![0, 3]),
        IntegerStorage::U64(vec![0, u64::MAX]),
    ];
    for storage in storages {
        let class = storage.class_name();
        let lhs = Tensor::new_integer(storage, vec![1, 2]).unwrap();
        let result = run_and(Value::Tensor(lhs), Value::Bool(true))
            .unwrap_or_else(|error| panic!("{class}: {error}"));
        assert!(
            matches!(
                &result,
                Value::LogicalArray(array)
                    if array.shape == vec![1, 2] && array.data == vec![0, 1]
            ),
            "{class}: unexpected result {result:?}"
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_char_arrays() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let lhs = CharArray::new("Run".chars().collect(), 1, 3).unwrap();
    let rhs = CharArray::new(vec!['R', 'u', '\0'], 1, 3).unwrap();
    let result = run_and(Value::CharArray(lhs), Value::CharArray(rhs)).expect("and char arrays");
    match result {
        Value::LogicalArray(arr) => {
            assert_eq!(arr.shape, vec![1, 3]);
            assert_eq!(arr.data, vec![1, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_treats_nan_as_true() {
    let result = run_and(Value::Num(f64::NAN), Value::Num(1.0)).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_complex_inputs() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let result = run_and(Value::Complex(0.0, 0.0), Value::Complex(0.0, 2.0)).unwrap();
    assert_eq!(result, Value::Bool(false));

    let result = run_and(Value::Complex(1.0, 0.0), Value::Complex(0.0, 2.0)).unwrap();
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_complex_and_character_extensions_are_mode_gated_before_dispatch() {
    assert_eq!(AND_EXTENSIONS[0].id, "and-complex-input");
    assert_eq!(AND_EXTENSIONS[1].id, "and-character-input");
    let complex_handle = runmat_accelerate_api::GpuTensorHandle {
        shape: vec![1, 1],
        device_id: 0,
        buffer_id: 9_347_001,
        descriptor: Default::default(),
    }
    .with_numeric_descriptor(
        runmat_accelerate_api::NumericElementType::F64,
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
    );
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    for complex in [
        Value::Complex(1.0, 1.0),
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 1.0)], vec![1, 1]).unwrap()),
        Value::GpuTensor(complex_handle.clone()),
    ] {
        let error = run_and(complex, Value::Bool(true)).expect_err("MATLAB mode rejects complex");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:AndComplexInputExtension")
        );
    }
    let chars = CharArray::new_row("A");
    let error = run_and(Value::CharArray(chars), Value::Bool(true))
        .expect_err("MATLAB mode rejects character arrays");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:AndCharacterInputExtension")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_size_mismatch_errors() {
    let lhs = Tensor::new(vec![1.0, 0.0, 2.0, 0.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![1.0, 0.0, 3.0], vec![3, 1]).unwrap();
    let err = run_and(Value::Tensor(lhs), Value::Tensor(rhs)).unwrap_err();
    assert_error_contains(&err, "size mismatch");
    assert_eq!(err.identifier(), AND_ERROR_SIZE_MISMATCH.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_rejects_unsupported_types() {
    let err = run_and(Value::String("runmat".into()), Value::Bool(true)).unwrap_err();
    assert_error_contains(&err, "unsupported input type");
    assert_eq!(err.identifier(), AND_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let a = provider.upload(&view).unwrap();
        let b = provider.upload(&view).unwrap();
        let result = run_and(Value::GpuTensor(a), Value::GpuTensor(b)).unwrap();
        let gathered = test_support::gather(result).unwrap();
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_gpu_supports_broadcast() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![0.0, 2.0, 0.0, 4.0], vec![4, 1]).unwrap();
        let rhs = Tensor::new(vec![1.0], vec![1, 1]).unwrap();

        let lhs_view = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        };
        let rhs_view = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        };

        let gpu_lhs = provider.upload(&lhs_view).expect("upload lhs");
        let gpu_rhs = provider.upload(&rhs_view).expect("upload rhs");

        let result = run_and(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs)).expect("and");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![4, 1]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 0.0, 1.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_resident_integer_fallback_is_exact_and_restores_residency() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new_integer(
            IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
            vec![3, 1],
        )
        .unwrap();
        let rhs = Tensor::new_integer(IntegerStorage::I8(vec![1, 0, -1]), vec![3, 1]).unwrap();
        let gpu_lhs = gpu_helpers::upload_tensor(provider, &lhs)
            .expect("upload lhs")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let gpu_rhs = gpu_helpers::upload_tensor(provider, &rhs)
            .expect("upload rhs")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);

        let result = run_and(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs))
            .expect("resident integer and");
        assert!(
            matches!(result, Value::GpuTensor(_)),
            "fallback must restore gpuArray residency"
        );
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![3, 1]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 0.0, 1.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn and_resident_complex_extension_gathers_and_restores_residency() {
    test_support::with_test_provider(|provider| {
        let lhs = ComplexTensor::new(vec![(0.0, 0.0), (0.0, 2.0)], vec![2, 1]).unwrap();
        let rhs = ComplexTensor::new(vec![(1.0, 0.0), (3.0, 0.0)], vec![2, 1]).unwrap();
        let gpu_lhs = gpu_helpers::upload_complex_tensor(provider, &lhs)
            .expect("upload lhs")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let gpu_rhs = gpu_helpers::upload_complex_tensor(provider, &rhs)
            .expect("upload rhs")
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);

        let result = run_and(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs))
            .expect("resident complex and");
        assert!(
            matches!(result, Value::GpuTensor(_)),
            "fallback must restore gpuArray residency"
        );
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn and_wgpu_matches_host_path() {
    let _accel_guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };

    let lhs = Tensor::new(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![1.0, 0.0, 3.0, 4.0], vec![2, 2]).unwrap();

    let cpu_value =
        run_and_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).expect("host and");
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

    let gpu_value = run_and(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs)).expect("gpu and");
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn and_wgpu_integer_broadcast_fallback_restores_logical_gpuarray() {
    let _accel_guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let lhs = Tensor::new_integer(
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993]),
        vec![2, 1],
    )
    .unwrap();
    let rhs = Tensor::new_integer(IntegerStorage::I8(vec![1, 0]), vec![1, 2]).unwrap();
    let gpu_lhs = gpu_helpers::upload_tensor(provider, &lhs)
        .expect("upload lhs")
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
    let gpu_rhs = gpu_helpers::upload_tensor(provider, &rhs)
        .expect("upload rhs")
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);

    let result = run_and(Value::GpuTensor(gpu_lhs), Value::GpuTensor(gpu_rhs))
        .expect("wgpu integer broadcast and");
    assert!(
        matches!(result, Value::GpuTensor(_)),
        "fallback must restore gpuArray residency"
    );
    let gathered = test_support::gather(result).expect("gather");
    assert_eq!(gathered.shape, vec![2, 2]);
    assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 0.0, 0.0]);
}
