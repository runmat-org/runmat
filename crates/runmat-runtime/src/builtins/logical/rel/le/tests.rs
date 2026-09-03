use super::*;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use futures::executor::block_on;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::AccelProvider;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::LE_ERROR_INVALID_INPUT;
use runmat_value::{CharArray, StringArray, Tensor};

fn run_le(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    block_on(super::le_builtin(lhs, rhs))
}

#[test]
fn le_dense_integer_arrays_read_exact_storage_without_mirror() {
    let lhs = Tensor::new_integer(
        runmat_value::IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]),
        vec![2, 1],
    )
    .expect("lhs");
    let rhs = Tensor::new_integer(
        runmat_value::IntegerStorage::I64(vec![0, 1, i64::MAX]),
        vec![1, 3],
    )
    .expect("rhs");

    let result = run_le(Value::Tensor(lhs), Value::Tensor(rhs)).expect("le");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 3]);
            assert_eq!(array.data, vec![1, 0, 1, 0, 1, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg(feature = "wgpu")]
fn run_le_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    comparison::evaluate_host(
        lhs,
        rhs,
        runmat_builtins::RelationalOperator::LessThanOrEqual,
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn le_scalar_true_for_equal_values() {
    let result = run_le(Value::Num(4.0), Value::Num(4.0)).expect("le");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn le_scalar_false() {
    let result = run_le(Value::Num(5.0), Value::Num(3.0)).expect("le");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn le_vector_broadcast() {
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![1, 4]).unwrap();
    let result = run_le(Value::Tensor(tensor), Value::Num(3.0)).expect("le");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 4]);
            assert_eq!(array.data, vec![1, 0, 1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn le_char_array_against_numeric() {
    let chars = CharArray::new(vec!['A', 'B', 'C'], 1, 3).unwrap();
    let tensor = Tensor::new(vec![65.0, 66.0, 67.0], vec![1, 3]).unwrap();
    let result = run_le(Value::CharArray(chars), Value::Tensor(tensor)).expect("le");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(array.data, vec![1, 1, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn le_string_array_against_scalar() {
    let array = StringArray::new(vec!["apple".into(), "carrot".into()], vec![1, 2]).unwrap();
    let result = run_le(Value::StringArray(array), Value::String("banana".into())).expect("le");
    match result {
        Value::LogicalArray(mask) => {
            assert_eq!(mask.shape, vec![1, 2]);
            assert_eq!(mask.data, vec![1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn le_string_numeric_error() {
    let err = run_le(Value::String("apple".into()), Value::Num(3.0)).expect_err("expected error");
    assert!(err.message().contains("mixing numeric and string"));
    assert_eq!(err.identifier(), LE_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn le_complex_compares_real_component() {
    let result = run_le(Value::Complex(2.0, 99.0), Value::Num(2.0)).expect("le");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn le_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![2.0, 4.0, 8.0], vec![1, 3]).unwrap();
        let view_l = HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        };
        let view_r = HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        };
        let handle_l = provider.upload(&view_l).expect("upload lhs");
        let handle_r = provider.upload(&view_r).expect("upload rhs");
        let result = run_le(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).expect("le");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 3]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0, 1.0]);
    });
}

#[test]
fn le_mixed_explicit_gpu_host_restores_logical_residency() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1, 4]),
            vec![2, 1],
        )
        .expect("lhs");
        let handle = gpu_helpers::upload_tensor(provider, &lhs).expect("upload");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let result =
            run_le(Value::GpuTensor(handle), Value::Num((1_u64 << 53) as f64)).expect("mixed le");
        let Value::GpuTensor(output) = result else {
            panic!("explicit gpuArray comparison must remain resident");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert!(runmat_accelerate_api::handle_is_logical(&output));
        let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather");
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn le_wgpu_matches_host() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let lhs = Tensor::new(vec![0.0, 2.0, 5.0, 7.0], vec![4, 1]).unwrap();
    let rhs = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![4, 1]).unwrap();
    let cpu = run_le_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();

    let view_l = HostTensorView {
        data: &lhs.materialize_f64(),
        shape: &lhs.shape,
    };
    let view_r = HostTensorView {
        data: &rhs.materialize_f64(),
        shape: &rhs.shape,
    };
    let handle_l = provider.upload(&view_l).expect("upload lhs");
    let handle_r = provider.upload(&view_r).expect("upload rhs");
    let gpu = run_le(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");

    match (cpu, gathered) {
        (Value::LogicalArray(host), tensor) => {
            assert_eq!(tensor.shape, host.shape);
            let expected: Vec<f64> = host
                .data
                .iter()
                .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                .collect();
            assert_eq!(tensor.materialize_f64(), expected);
        }
        (Value::Bool(host_flag), tensor) => {
            assert_eq!(tensor.shape, vec![1, 1]);
            let expected = if host_flag { 1.0 } else { 0.0 };
            assert_eq!(tensor.materialize_f64(), vec![expected]);
        }
        other => panic!("unexpected output combination: {other:?}"),
    }
}
