use super::*;
use crate::builtins::common::gpu_helpers;
#[cfg(feature = "wgpu")]
use crate::builtins::common::tensor;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::ProviderPrecision;
use runmat_builtins::{EQ_ERROR_INVALID_INPUT, EQ_ERROR_SIZE_MISMATCH};
use runmat_value::{CharArray, LogicalArray, StringArray, Tensor};
use runmat_value::{
    ComplexTensor, HandleRef, IntegerComplexStorage, IntegerStorage, Listener, SymbolicArray,
    SymbolicExpr,
};

fn run_eq(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    block_on(super::eq_builtin(lhs, rhs))
}

#[test]
fn eq_scalar_complex_reads_all_typed_integer_classes_without_f64_mirrors() {
    let cases = [
        (IntegerStorage::I8(vec![-7]), -7.0, true),
        (IntegerStorage::I16(vec![-300]), -300.0, true),
        (IntegerStorage::I32(vec![-70_000]), -70_000.0, true),
        (
            IntegerStorage::I64(vec![-9_007_199_254_740_991]),
            -9_007_199_254_740_991.0,
            true,
        ),
        (IntegerStorage::U8(vec![7]), 7.0, true),
        (IntegerStorage::U16(vec![300]), 300.0, true),
        (IntegerStorage::U32(vec![70_000]), 70_000.0, true),
        (
            IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            (1_u64 << 53) as f64,
            false,
        ),
    ];

    for (storage, real, expected) in cases {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("integer scalar");
        assert_eq!(
            run_eq(Value::Tensor(tensor), Value::Complex(real, 0.0)).expect("eq"),
            Value::Bool(expected)
        );
    }
}

#[test]
fn eq_dense_integer_arrays_read_exact_storage_without_mirror() {
    let lhs = Tensor::new_integer(IntegerStorage::U64(vec![0, (1_u64 << 53) + 1]), vec![2, 1])
        .expect("lhs");
    let rhs =
        Tensor::new_integer(IntegerStorage::I64(vec![0, 1, i64::MAX]), vec![1, 3]).expect("rhs");

    let result = run_eq(Value::Tensor(lhs), Value::Tensor(rhs)).expect("eq");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 3]);
            assert_eq!(array.data, vec![1, 0, 0, 0, 0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg(feature = "wgpu")]
fn run_eq_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    comparison::evaluate_host(lhs, rhs, runmat_builtins::RelationalOperator::Equal)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_scalar_true() {
    let result = run_eq(Value::Num(5.0), Value::Num(5.0)).expect("eq");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_scalar_false() {
    let result = run_eq(Value::Num(5.0), Value::Num(4.0)).expect("eq");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_vector_broadcast() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 2.0], vec![1, 4]).unwrap();
    let result = run_eq(Value::Tensor(tensor), Value::Num(2.0)).expect("eq");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 4]);
            assert_eq!(array.data, vec![0, 1, 0, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_char_array_against_numeric() {
    let char_array = CharArray::new(vec!['A', 'B', 'A'], 1, 3).unwrap();
    let tensor = Tensor::new(vec![65.0, 66.0, 65.0], vec![1, 3]).unwrap();
    let result = run_eq(Value::CharArray(char_array), Value::Tensor(tensor)).expect("eq");
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
fn eq_string_array_broadcast() {
    let sa = StringArray::new(vec!["red".into(), "blue".into()], vec![1, 2]).unwrap();
    let result = run_eq(Value::StringArray(sa), Value::String("red".into())).expect("eq");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 2]);
            assert_eq!(array.data, vec![1, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[test]
fn eq_symbolic_scalar_builds_equation() {
    let applied = SymbolicExpr::function_call("Y", vec![SymbolicExpr::constant(0.0)]);

    let result = run_eq(Value::Symbolic(applied), Value::Num(0.0)).expect("eq");

    assert_eq!(result.to_string(), "Y(0) == 0");
}

#[test]
fn eq_numeric_scalar_with_symbolic_scalar_builds_equation() {
    let result = run_eq(
        Value::Num(0.0),
        Value::Symbolic(SymbolicExpr::variable("x")),
    )
    .expect("eq");

    assert_eq!(result.to_string(), "0 == x");
}

#[test]
fn eq_symbolic_scalar_with_symbolic_scalar_builds_equation() {
    let result = run_eq(
        Value::Symbolic(SymbolicExpr::variable("x")),
        Value::Symbolic(SymbolicExpr::variable("y")),
    )
    .expect("eq");

    assert_eq!(result.to_string(), "x == y");
}

#[test]
fn eq_symbolic_array_with_scalar_builds_equations() {
    let array = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![1, 2],
    )
    .unwrap();

    let result = run_eq(Value::SymbolicArray(array), Value::Num(0.0)).expect("eq");

    match result {
        Value::SymbolicArray(array) => {
            assert_eq!(array.shape, vec![1, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x == 0", "y == 0"]
            );
        }
        other => panic!("expected symbolic array, got {other:?}"),
    }
}

#[test]
fn eq_compatible_symbolic_arrays_build_equations() {
    let lhs = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![1, 2],
    )
    .unwrap();
    let rhs = SymbolicArray::new(
        vec![SymbolicExpr::constant(1.0), SymbolicExpr::constant(2.0)],
        vec![1, 2],
    )
    .unwrap();

    let result = run_eq(Value::SymbolicArray(lhs), Value::SymbolicArray(rhs)).expect("eq");

    match result {
        Value::SymbolicArray(array) => {
            assert_eq!(array.shape, vec![1, 2]);
            assert_eq!(
                array
                    .data
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>(),
                vec!["x == 1", "y == 2"]
            );
        }
        other => panic!("expected symbolic array, got {other:?}"),
    }
}

#[test]
fn eq_symbolic_array_shape_mismatch_errors() {
    let lhs = SymbolicArray::new(
        vec![SymbolicExpr::variable("x"), SymbolicExpr::variable("y")],
        vec![1, 2],
    )
    .unwrap();
    let rhs = SymbolicArray::new(
        vec![
            SymbolicExpr::constant(1.0),
            SymbolicExpr::constant(2.0),
            SymbolicExpr::constant(3.0),
        ],
        vec![1, 3],
    )
    .unwrap();

    let err = run_eq(Value::SymbolicArray(lhs), Value::SymbolicArray(rhs))
        .expect_err("shape mismatch should fail");

    assert_eq!(err.identifier(), EQ_ERROR_SIZE_MISMATCH.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_handle_identity() {
    let ptr = runmat_gc::gc_allocate(Value::Num(1.0)).expect("gc allocation");
    let handle = HandleRef {
        class_name: "Dummy".into(),
        target: ptr,
        valid: true,
    };
    let a = Value::HandleObject(handle.clone());
    let b = Value::HandleObject(handle.clone());
    assert_eq!(run_eq(a.clone(), b.clone()).unwrap(), Value::Bool(true));

    let other_ptr = runmat_gc::gc_allocate(Value::Num(2.0)).expect("gc allocation");
    let other_handle = HandleRef {
        class_name: "Dummy".into(),
        target: other_ptr,
        valid: true,
    };
    let other = Value::HandleObject(other_handle);
    assert_eq!(run_eq(a, other).unwrap(), Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_listener_identity_is_disjoint_from_target_identity() {
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
        run_eq(
            Value::Listener(listener_a.clone()),
            Value::Listener(listener_b)
        )
        .unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        run_eq(Value::Listener(listener_a), Value::HandleObject(handle)).unwrap(),
        Value::Bool(false)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let a = provider.upload(&view).expect("upload");
        let b = provider.upload(&view).expect("upload");
        let result = run_eq(Value::GpuTensor(a), Value::GpuTensor(b)).expect("gpu eq succeeds");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![3, 1]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, 1.0, 1.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_numeric_and_string_error() {
    let err = run_eq(Value::Num(1.0), Value::String("a".into())).unwrap_err();
    assert!(err.message().contains("mixing numeric and string inputs"));
    assert_eq!(err.identifier(), EQ_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eq_complex_and_numeric() {
    let complex = Value::Complex(2.0, 0.0);
    let numeric = Value::Num(2.0);
    assert_eq!(run_eq(complex, numeric).unwrap(), Value::Bool(true));
}

#[test]
fn eq_typed_complex_integer_compares_exact_storage() {
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
        run_eq(Value::ComplexTensor(tensor), Value::ComplexTensor(rhs)).unwrap(),
        Value::LogicalArray(
            LogicalArray::new(vec![1, 0, 0, 0, 0, 1], vec![2, 3]).expect("logical result")
        )
    );
}

#[test]
fn eq_typed_complex_integer_real_comparison_requires_zero_imaginary_part() {
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
        run_eq(
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
            LogicalArray::new(vec![1, 0, 0, 0], vec![2, 2]).expect("logical result")
        )
    );
}

#[test]
fn eq_resident_integer_with_host_scalar_falls_back_exactly() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![(1_u64 << 53) + 1, u64::MAX]),
            vec![2, 1],
        )
        .expect("integer tensor");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("integer upload");
        let result = run_eq(
            Value::GpuTensor(handle.clone()),
            Value::Int(runmat_value::IntValue::U64((1_u64 << 53) + 1)),
        )
        .expect("exact fallback");
        let gathered = test_support::gather(result).expect("gather logical result");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0]);
        provider.free(&handle).expect("free input");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn eq_wgpu_matches_host() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let cpu = run_eq_host(Value::Tensor(tensor.clone()), Value::Tensor(tensor.clone())).unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let provider = runmat_accelerate_api::provider().unwrap();
    let a = provider.upload(&view).unwrap();
    let b = provider.upload(&view).unwrap();
    let gpu = run_eq(Value::GpuTensor(a), Value::GpuTensor(b)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match (cpu, gathered) {
        (Value::LogicalArray(expected), actual) => {
            assert_eq!(actual.shape, expected.shape);
            let tol = match provider.precision() {
                ProviderPrecision::F64 => 1e-12,
                ProviderPrecision::F32 => 1e-5,
            };
            for (idx, value) in actual.materialize_f64().iter().enumerate() {
                let expected_val = expected.data[idx] as f64;
                assert!((value - expected_val).abs() <= tol);
            }
        }
        (Value::Bool(flag), actual) => {
            assert_eq!(tensor::element_count(&actual.shape), 1);
            assert_eq!(actual.materialize_f64()[0] != 0.0, flag);
        }
        other => panic!("unexpected comparison result {other:?}"),
    }
}
