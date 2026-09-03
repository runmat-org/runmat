use super::*;
#[cfg(feature = "wgpu")]
use crate::builtins::common::tensor;
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

fn run_not(value: Value) -> BuiltinResult<Value> {
    block_on(super::not_builtin(value))
}

#[cfg(feature = "wgpu")]
fn run_not_host(value: Value) -> BuiltinResult<Value> {
    block_on(not_host(value))
}
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::{CharArray, ComplexTensor, IntValue, IntegerStorage, LogicalArray, Tensor};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_of_booleans() {
    assert_eq!(run_not(Value::Bool(true)).unwrap(), Value::Bool(false));
    assert_eq!(run_not(Value::Bool(false)).unwrap(), Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_numeric_array() {
    let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]).unwrap();
    let result = run_not(Value::Tensor(tensor)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 2]);
            assert_eq!(array.data, vec![1, 0, 0, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_complex_scalar() {
    let result = run_not(Value::Complex(0.0, 0.0)).expect("not complex zero should succeed");
    assert_eq!(result, Value::Bool(true));

    let result = run_not(Value::Complex(1.0, 0.0)).expect("not complex nonzero should succeed");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_nan_yields_false() {
    let result = run_not(Value::Num(f64::NAN)).unwrap();
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_char_array() {
    let chars = CharArray::new_row("A\0C");
    let result = run_not(Value::CharArray(chars)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(array.data, vec![0, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 1.0, 0.0, 2.0], vec![2, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = run_not(Value::GpuTensor(handle)).expect("not on gpu");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 1.0, 0.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_accepts_int_inputs() {
    let value = Value::Int(IntValue::I32(0));
    assert_eq!(run_not(value).unwrap(), Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_reads_typed_integer_storage_exactly_for_truth_values() {
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![0, 9_007_199_254_740_993, u64::MAX]),
        vec![1, 3],
    )
    .unwrap();
    let result = run_not(Value::Tensor(tensor)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(array.data, vec![1, 0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }

    assert_eq!(
        run_not(Value::Int(IntValue::U64(u64::MAX))).unwrap(),
        Value::Bool(false)
    );
}

#[test]
fn not_maps_table_variables_and_preserves_the_container() {
    let table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 2.0], vec![2, 1]).unwrap()),
        ],
    )
    .unwrap();

    let Value::Object(output) = run_not(table).expect("table not") else {
        panic!("expected table output")
    };
    assert_eq!(output.class_name, runmat_types::standard::TABLE);
    let variables = crate::builtins::table::table_variables(&output).unwrap();
    assert!(matches!(
        &variables.fields["A"],
        Value::LogicalArray(values) if values.data == vec![0, 1]
    ));
    assert!(matches!(
        &variables.fields["B"],
        Value::LogicalArray(values) if values.data == vec![1, 0]
    ));
}

#[test]
fn not_accepts_every_integer_class_without_floating_truth_conversion() {
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
        let result = run_not(Value::Tensor(input)).expect("not");
        let Value::LogicalArray(output) = result else {
            panic!("expected logical array");
        };
        assert_eq!(output.data, vec![1, 0]);
    }
}

#[test]
fn not_explicit_resident_integer_fallback_is_exact_and_stays_resident() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![0, (1_u64 << 53) + 1, u64::MAX]),
            vec![1, 3],
        )
        .expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let result = run_not(Value::GpuTensor(handle)).expect("not");
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
fn not_tensor_scalar_returns_bool() {
    let tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
    assert_eq!(run_not(Value::Tensor(tensor)).unwrap(), Value::Bool(false));

    let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
    assert_eq!(run_not(Value::Tensor(tensor)).unwrap(), Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_empty_tensor_preserves_shape() {
    let tensor = Tensor::new(Vec::<f64>::new(), vec![0, 3]).unwrap();
    let result = run_not(Value::Tensor(tensor)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![0, 3]);
            assert!(array.data.is_empty());
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_complex_tensor() {
    let tensor = ComplexTensor::new(vec![(0.0, 0.0), (1.0, 0.0), (0.0, -2.0)], vec![3, 1]).unwrap();
    let result = run_not(Value::ComplexTensor(tensor)).unwrap();
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![3, 1]);
            assert_eq!(array.data, vec![1, 0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_logical_array_flips_bits() {
    let array = LogicalArray::new(vec![1, 0, 1, 1], vec![2, 2]).unwrap();
    let result = run_not(Value::LogicalArray(array)).unwrap();
    match result {
        Value::LogicalArray(out) => {
            assert_eq!(out.shape, vec![2, 2]);
            assert_eq!(out.data, vec![0, 1, 0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn not_rejects_string_input() {
    let err = run_not(Value::String("abc".into())).unwrap_err();
    assert_error_contains(&err, "unsupported input type");
    assert_eq!(err.identifier(), NOT_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn not_wgpu_integer_handle_uses_exact_fallback_and_preserves_explicit_residency() {
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
    let result = run_not(Value::GpuTensor(handle)).expect("not");
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
fn not_wgpu_matches_host_path() {
    let _accel_guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let tensor = Tensor::new(vec![0.0, 3.0, 0.0, -1.0], vec![2, 2]).unwrap();
    let cpu = run_not_host(Value::Tensor(tensor.clone())).unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).unwrap();
    let gpu = run_not(Value::GpuTensor(handle)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    let cpu_tensor = tensor::value_to_tensor(&cpu).expect("cpu tensor");
    assert_eq!(gathered.shape, cpu_tensor.shape);
    let tol = match provider.precision() {
        ProviderPrecision::F64 => 1e-12,
        ProviderPrecision::F32 => 1e-5,
    };
    for (expected, actual) in cpu_tensor
        .materialize_f64()
        .iter()
        .zip(gathered.materialize_f64().iter())
    {
        assert!((*expected - *actual).abs() < tol, "{expected} vs {actual}");
    }
}
