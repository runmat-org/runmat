use super::*;
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::GE_INTEGER_CAPABILITIES;
use runmat_builtins::{
    BuiltinIntegerComputationDomain, BuiltinIntegerOutputClassRule, BuiltinIntegerOverloadKind,
    GE_ERROR_INVALID_INPUT,
};
use runmat_value::{CharArray, StringArray, Tensor};

fn run_ge(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    block_on(super::ge_builtin(lhs, rhs))
}

#[test]
fn ge_dense_integer_arrays_read_exact_storage_without_mirror() {
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

    let result = run_ge(Value::Tensor(lhs), Value::Tensor(rhs)).expect("ge");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![2, 3]);
            assert_eq!(array.data, vec![1, 1, 0, 1, 0, 0]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[test]
fn ge_integer_contract_is_exact_broadcasting_to_logical() {
    assert_eq!(GE_INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(GE_INTEGER_CAPABILITIES[0].inputs.len(), 2);
    assert!(GE_INTEGER_CAPABILITIES[0]
        .inputs
        .iter()
        .all(|input| input.classes.len() == 8));
    assert_eq!(
        GE_INTEGER_CAPABILITIES[0].computation_domain,
        BuiltinIntegerComputationDomain::Predicate
    );
    assert_eq!(
        GE_INTEGER_CAPABILITIES[0].output_class,
        BuiltinIntegerOutputClassRule::Logical
    );
    assert_eq!(
        GE_INTEGER_CAPABILITIES[0].overload,
        BuiltinIntegerOverloadKind::BroadcastCompatible
    );
}

#[test]
fn ge_gpu_wide_integer_scalar_fallback_is_exact_owner_resident_logical() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_992, 9_007_199_254_740_993]),
            vec![1, 2],
        )
        .expect("wide integer lhs");
        let handle = gpu_helpers::upload_tensor(provider, &lhs).expect("upload exact lhs");
        let result = run_ge(
            Value::GpuTensor(handle),
            Value::Int(runmat_value::IntValue::U64(9_007_199_254_740_993)),
        )
        .expect("resident exact ge");
        let Value::GpuTensor(result_handle) = &result else {
            panic!("expected resident logical result");
        };
        assert!(runmat_accelerate_api::handle_is_logical(result_handle));
        assert!(std::ptr::eq(
            runmat_accelerate_api::provider_for_handle(result_handle).expect("result owner"),
            provider
        ));
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.shape, vec![1, 2]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0]);
    });
}

#[test]
fn ge_resident_single_does_not_round_a_double_scalar_for_comparison() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::from_f32(vec![16_777_216.0], vec![1, 1]).expect("single lhs");
        let handle = gpu_helpers::upload_tensor(provider, &lhs).expect("upload single lhs");
        let result =
            run_ge(Value::GpuTensor(handle), Value::Num(16_777_217.0)).expect("mixed-precision ge");
        let Value::GpuTensor(result_handle) = &result else {
            panic!("expected resident logical result");
        };
        assert!(runmat_accelerate_api::handle_is_logical(result_handle));
        assert!(std::ptr::eq(
            runmat_accelerate_api::provider_for_handle(result_handle).expect("result owner"),
            provider
        ));
        let gathered = test_support::gather(result).expect("gather result");
        assert_eq!(gathered.materialize_f64(), vec![0.0]);
    });
}

#[cfg(feature = "wgpu")]
fn run_ge_host(lhs: Value, rhs: Value) -> crate::BuiltinResult<Value> {
    comparison::evaluate_host(
        lhs,
        rhs,
        runmat_builtins::RelationalOperator::GreaterThanOrEqual,
    )
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ge_scalar_true_for_equal_values() {
    let result = run_ge(Value::Num(5.0), Value::Num(5.0)).expect("ge");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ge_scalar_false() {
    let result = run_ge(Value::Num(2.0), Value::Num(3.0)).expect("ge");
    assert_eq!(result, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ge_vector_broadcast() {
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0], vec![1, 4]).unwrap();
    let result = run_ge(Value::Tensor(tensor), Value::Num(4.0)).expect("ge");
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
fn ge_vector_including_equal_values() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let result = run_ge(Value::Tensor(tensor), Value::Num(2.0)).expect("ge");
    match result {
        Value::LogicalArray(array) => {
            assert_eq!(array.shape, vec![1, 3]);
            assert_eq!(array.data, vec![0, 1, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ge_char_array_against_numeric() {
    let chars = CharArray::new(vec!['A', 'B', 'C'], 1, 3).unwrap();
    let tensor = Tensor::new(vec![65.0, 66.0, 66.0], vec![1, 3]).unwrap();
    let result = run_ge(Value::CharArray(chars), Value::Tensor(tensor)).expect("ge");
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
fn ge_string_array_against_scalar() {
    let array = StringArray::new(vec!["apple".into(), "banana".into()], vec![1, 2]).unwrap();
    let result = run_ge(Value::StringArray(array), Value::String("banana".into())).expect("ge");
    match result {
        Value::LogicalArray(mask) => {
            assert_eq!(mask.shape, vec![1, 2]);
            assert_eq!(mask.data, vec![0, 1]);
        }
        other => panic!("expected logical array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ge_string_numeric_error() {
    let err = run_ge(Value::String("apple".into()), Value::Num(3.0)).expect_err("expected error");
    assert!(err.message().contains("mixing numeric and string"));
    assert_eq!(err.identifier(), GE_ERROR_INVALID_INPUT.identifier);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ge_complex_compares_real_component() {
    let result = run_ge(Value::Complex(2.0, -99.0), Value::Num(2.0)).expect("ge");
    assert_eq!(result, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ge_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![1.0, 4.0, 6.0], vec![1, 3]).unwrap();
        let rhs = Tensor::new(vec![1.0, 5.0, 6.0], vec![1, 3]).unwrap();
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
        let result = run_ge(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).expect("ge");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![1, 3]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, 0.0, 1.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn ge_wgpu_matches_host() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let lhs = Tensor::new(vec![0.0, 2.0, 5.0, 8.0], vec![4, 1]).unwrap();
    let rhs = Tensor::new(vec![0.0, 2.5, 5.0, 9.0], vec![4, 1]).unwrap();
    let cpu = run_ge_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();

    let view_l = HostTensorView {
        data: &lhs.materialize_f64(),
        shape: &lhs.shape,
    };
    let view_r = HostTensorView {
        data: &rhs.materialize_f64(),
        shape: &rhs.shape,
    };
    let provider = runmat_accelerate_api::provider().expect("provider");
    let handle_l = provider.upload(&view_l).expect("upload lhs");
    let handle_r = provider.upload(&view_r).expect("upload rhs");
    let gpu = run_ge(Value::GpuTensor(handle_l), Value::GpuTensor(handle_r)).unwrap();
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
