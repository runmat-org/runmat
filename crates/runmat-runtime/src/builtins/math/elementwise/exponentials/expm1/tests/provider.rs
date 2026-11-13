use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_builtins::EXPM1_ERROR_INVALID_INPUT;
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn resident_integer_checks_exactness_before_floating_conversion() {
    test_support::with_test_provider(|provider| {
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let error = call(Value::GpuTensor(handle)).expect_err("inexact integer");
        assert_eq!(error.identifier(), EXPM1_ERROR_INVALID_INPUT.identifier);
    });
}

#[test]
fn provider_result_remains_resident_and_preserves_tiny_values() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![0.0, 1.0e-12, 1.0], vec![3, 1]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let output = call(Value::GpuTensor(handle)).expect("provider expm1");
        assert!(matches!(output, Value::GpuTensor(_)));
        let gathered = test_support::gather(output).expect("gather");
        for (actual, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(input.materialize_f64().iter().map(|value| value.exp_m1()))
        {
            assert!((*actual - expected).abs() < 1e-12);
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn expm1_wgpu_matches_cpu_elementwise() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let input = Tensor::new(vec![0.0, 0.25, 1.0], vec![3, 1]).expect("input");
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
    let output = call(Value::GpuTensor(handle)).expect("expm1");
    let gathered = test_support::gather(output).expect("gather");
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(input.materialize_f64().iter().map(|value| value.exp_m1()))
    {
        assert!((*actual - expected).abs() < 1e-5);
    }
}
