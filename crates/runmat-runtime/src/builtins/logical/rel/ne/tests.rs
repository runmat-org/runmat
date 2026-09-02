use super::*;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::NE_ERROR_INVALID_INPUT;
use runmat_value::{CharArray, LogicalArray, StringArray, Tensor};
use runmat_value::{ComplexTensor, HandleRef, IntegerComplexStorage, IntegerStorage, Listener};

fn run_ne(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    block_on(super::ne_builtin(lhs, rhs))
}

#[test]
fn ne_scalar_complex_reads_all_typed_integer_classes_without_f64_mirrors() {
    let cases = [
        (IntegerStorage::I8(vec![-7]), -7.0, false),
        (IntegerStorage::I16(vec![-300]), -300.0, false),
        (IntegerStorage::I32(vec![-70_000]), -70_000.0, false),
        (
            IntegerStorage::I64(vec![-9_007_199_254_740_991]),
            -9_007_199_254_740_991.0,
            false,
        ),
        (IntegerStorage::U8(vec![7]), 7.0, false),
        (IntegerStorage::U16(vec![300]), 300.0, false),
        (IntegerStorage::U32(vec![70_000]), 70_000.0, false),
        (
            IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            (1_u64 << 53) as f64,
            true,
        ),
    ];

    for (storage, real, expected) in cases {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer scalar");
        assert_eq!(
            run_ne(Value::Tensor(tensor), Value::Complex(real, 0.0)).expect("ne"),
            Value::Bool(expected)
        );
    }
}

#[test]
fn ne_dense_integer_arrays_read_exact_storage_without_mirror() {
    let lhs = Tensor::new_integer(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), vec![2, 1])
        .expect("lhs");
    let rhs =
        Tensor::new_integer(IntegerStorage::I64(vec![0, 1, i64::MAX]), vec![1, 3]).expect("rhs");

    let result = run_ne(Value::Tensor(lhs), Value::Tensor(rhs)).expect("ne");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 3]);
            assert_eq!(array.data, vec![0, 1, 1, 1, 1, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg(feature = "wgpu")]
fn run_ne_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    block_on(ne_host(lhs, rhs))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_scalar_true() {
    let result = run_ne(Value::Num(5.0), Value::Num(4.0)).expect("ne");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_scalar_false() {
    let result = run_ne(Value::Num(5.0), Value::Num(5.0)).expect("ne");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_vector_broadcast() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 2.0], vec![1, 4]).unwrap();
    let result = run_ne(Value::Tensor(tensor), Value::Num(2.0)).expect("ne");
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
fn ne_char_array_against_numeric() {
    let char_array = CharArray::new(vec!['A', 'B', 'A'], 1, 3).unwrap();
    let tensor = Tensor::new(vec![65.0, 66.0, 65.0], vec![1, 3]).unwrap();
    let result = run_ne(Value::CharArray(char_array), Value::Tensor(tensor)).expect("ne");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(array.data, vec![0, 0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_string_array_broadcast() {
    let sa = StringArray::new(vec!["red".into(), "blue".into()], vec![1, 2]).unwrap();
    let result = run_ne(Value::StringArray(sa), Value::String("red".into())).expect("ne");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 2]);
            assert_eq!(array.data, vec![0, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_handle_identity() {
    let gc = runmat_gc::gc_allocate(Value::Num(1.0)).expect("gc allocation");
    let handle = HandleRef {
        class_name: "TestHandle".into(),
        target: gc,
        valid: true,
    };
    let lhs = Value::HandleObject(handle.clone());
    let rhs = Value::HandleObject(handle);
    let result = run_ne(lhs, rhs).expect("ne");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_handle_difference() {
    let handle_a = HandleRef {
        class_name: "TestHandle".into(),
        target: runmat_gc::gc_allocate(Value::Num(1.0)).expect("gc allocation"),
        valid: true,
    };
    let handle_b = HandleRef {
        class_name: "TestHandle".into(),
        target: runmat_gc::gc_allocate(Value::Num(2.0)).expect("gc allocation"),
        valid: true,
    };
    let result = run_ne(Value::HandleObject(handle_a), Value::HandleObject(handle_b)).expect("ne");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_listener_identity_is_disjoint_from_target_identity() {
    let target = runmat_gc::gc_allocate(Value::Num(1.0)).expect("gc target");
    let callback =
        runmat_gc::gc_allocate(Value::FunctionHandle("cb".to_string())).expect("gc callback");
    let handle = HandleRef {
        class_name: "EventTarget".into(),
        target,
        valid: true,
    };
    let listener_a = Listener {
        id: 101,
        target,
        target_class_name: "EventTarget".into(),
        event_name: "Changed".into(),
        callback,
        enabled: true,
        valid: true,
    };
    let listener_b = Listener {
        id: 102,
        target,
        target_class_name: "EventTarget".into(),
        event_name: "Changed".into(),
        callback,
        enabled: true,
        valid: true,
    };

    assert_eq!(
        run_ne(
            Value::Listener(listener_a.clone()),
            Value::Listener(listener_b)
        )
        .unwrap(),
        Value::Bool(true)
    );
    assert_eq!(
        run_ne(Value::Listener(listener_a), Value::HandleObject(handle)).unwrap(),
        Value::Bool(true)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_mixed_numeric_string_error() {
    let err = run_ne(Value::Num(1.0), Value::String("a".into())).unwrap_err();
    assert!(err.message().contains("mixing numeric and string inputs"));
    assert_eq!(err.identifier(), NE_ERROR_INVALID_INPUT.identifier);
}

#[test]
fn ne_typed_complex_integer_compares_exact_storage() {
    let tensor = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
            IntegerStorage::U64(vec![0, 7]),
        )
        .expect("matching components"),
        vec![2, 1],
    )
    .expect("typed complex");
    let rhs = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1_u64 << 63, u64::MAX, u64::MAX]),
            IntegerStorage::U64(vec![0, 0, 7]),
        )
        .expect("matching components"),
        vec![1, 3],
    )
    .expect("typed complex rhs");

    assert_eq!(
        run_ne(Value::ComplexTensor(tensor), Value::ComplexTensor(rhs)).unwrap(),
        Value::LogicalArray(
            LogicalArray::new(vec![0, 1, 1, 1, 1, 0], vec![2, 3]).expect("logical result")
        )
    );
}

#[test]
fn ne_typed_complex_integer_real_comparison_requires_zero_imaginary_part() {
    let tensor = ComplexTensor::new_integer(
        IntegerComplexStorage::new(
            IntegerStorage::U64(vec![1_u64 << 63, (1_u64 << 63) + 1]),
            IntegerStorage::U64(vec![0, 1]),
        )
        .expect("matching components"),
        vec![2, 1],
    )
    .expect("typed complex");

    assert_eq!(
        run_ne(
            Value::ComplexTensor(tensor),
            Value::Tensor(
                Tensor::new_integer(
                    IntegerStorage::U64(vec![1_u64 << 63, (1_u64 << 63) + 1]),
                    vec![1, 2],
                )
                .expect("integer tensor")
            ),
        )
        .unwrap(),
        Value::LogicalArray(
            LogicalArray::new(vec![0, 1, 1, 1], vec![2, 2]).expect("logical result")
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![2, 2]).unwrap();
        let tensor_b = Tensor::new(vec![1.0, 0.0, 3.0, 5.0], vec![2, 2]).unwrap();
        let view_a = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let view_b = HostTensorView {
            data: &tensor_b.materialize_f64(),
            shape: &tensor_b.shape,
        };
        let h_a = provider.upload(&view_a).expect("upload a");
        let h_b = provider.upload(&view_b).expect("upload b");
        let result = run_ne(Value::GpuTensor(h_a), Value::GpuTensor(h_b)).expect("ne");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 1.0, 0.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ne_gpu_falls_back_to_host_when_only_one_tensor() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = run_ne(Value::GpuTensor(handle), Value::Num(2.0)).expect("ne");
        let gathered = test_support::gather(result).expect("gather logical result");
        assert_eq!(gathered.shape, vec![3, 1]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 1.0]);
    });
}

#[test]
fn ne_explicit_resident_wide_integer_fallback_stays_logical_and_resident() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            IntegerStorage::U64(vec![0, (1_u64 << 53) + 1, u64::MAX]),
            vec![1, 3],
        )
        .expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload integer");
        let handle = handle.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
        let result = run_ne(Value::GpuTensor(handle), Value::Num(0.0)).expect("ne");
        let Value::GpuTensor(output) = &result else {
            panic!("explicit gpuArray result must remain resident");
        };
        assert!(runmat_accelerate_api::handle_is_explicit(output));
        assert!(runmat_accelerate_api::handle_is_logical(output));
        assert_eq!(
            test_support::gather(result)
                .expect("gather")
                .materialize_f64(),
            vec![0.0, 1.0, 1.0]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn ne_wgpu_compares_wide_integer_handles_exactly_and_preserves_explicit_residency() {
    let _accel_guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU provider");
    let lhs = Tensor::new_integer(
        IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
        vec![1, 2],
    )
    .expect("lhs");
    let rhs = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 53, u64::MAX]), vec![1, 2])
        .expect("rhs");
    let lhs = gpu_helpers::upload_tensor(provider, &lhs).expect("upload lhs");
    let rhs = gpu_helpers::upload_tensor(provider, &rhs).expect("upload rhs");
    let lhs = lhs.with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit);
    let result = run_ne(Value::GpuTensor(lhs), Value::GpuTensor(rhs)).expect("ne");
    let Value::GpuTensor(output) = &result else {
        panic!("resident comparison must remain resident");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(output));
    assert_eq!(
        test_support::gather(result)
            .expect("gather")
            .materialize_f64(),
        vec![1.0, 0.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn ne_wgpu_matches_host() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![1.0, 0.0, 3.0, 5.0], vec![2, 2]).unwrap();
    let cpu = run_ne_host(Value::Tensor(a.clone()), Value::Tensor(b.clone())).unwrap();
    let view_a = HostTensorView {
        data: &a.materialize_f64(),
        shape: &a.shape,
    };
    let view_b = HostTensorView {
        data: &b.materialize_f64(),
        shape: &b.shape,
    };
    let provider = runmat_accelerate_api::provider().expect("provider");
    let h_a = provider.upload(&view_a).unwrap();
    let h_b = provider.upload(&view_b).unwrap();
    let gpu = run_ne(Value::GpuTensor(h_a), Value::GpuTensor(h_b)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match (cpu, gathered) {
        (Value::LogicalArray(cp), gt) => {
            assert_eq!(gt.shape, cp.shape);
            let tol = match provider.precision() {
                ProviderPrecision::F64 => 1e-12,
                ProviderPrecision::F32 => 1e-5,
            };
            for (a, b) in gt.materialize_f64().iter().zip(cp.data.iter()) {
                let diff = *a - f64::from(*b);
                assert!(
                    diff.abs() < tol,
                    "mismatch between GPU and CPU logical results"
                );
            }
        }
        _ => panic!("unexpected result variants"),
    }
}
