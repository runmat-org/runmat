use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_builtins::{LOG2_ERROR_INVALID_INPUT, LOG2_ERROR_PROVIDER_OWNERSHIP};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn provider_result_matches_host_across_device_or_fallback_execution() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let output = call(Value::GpuTensor(handle)).expect("provider log2");
        let gathered = test_support::gather(output).expect("gather");
        assert_eq!(gathered.materialize_f64(), &[0.0, 1.0, 2.0]);
    });
}

#[test]
fn resident_integer_checks_exactness_before_floating_conversion() {
    test_support::with_test_provider(|provider| {
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        assert_eq!(
            call(Value::GpuTensor(handle))
                .expect_err("inexact integer")
                .identifier(),
            LOG2_ERROR_INVALID_INPUT.identifier
        );
    });
}

#[test]
fn missing_owner_error_is_catalog_owned_and_terminal() {
    let error = super::super::super::errors::missing_provider(
        super::super::super::operation::LogarithmOperation::Binary,
    );
    assert_eq!(error.message(), LOG2_ERROR_PROVIDER_OWNERSHIP.message);
    assert_eq!(error.identifier(), LOG2_ERROR_PROVIDER_OWNERSHIP.identifier);
    assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
}

#[test]
fn negative_automatic_input_returns_host_complex_fallback() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![-4.0, -8.0], vec![2, 1]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let Value::ComplexTensor(output) = call(Value::GpuTensor(handle)).expect("log2") else {
            panic!("expected complex tensor")
        };
        assert_eq!(output.shape, vec![2, 1]);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn log2_wgpu_matches_cpu_elementwise() {
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
    let input = Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).expect("input");
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
    let output = call(Value::GpuTensor(handle)).expect("log2");
    let gathered = test_support::gather(output).expect("gather");
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(input.materialize_f64().iter().map(|value| value.log2()))
    {
        assert!((*actual - expected).abs() < 1e-5);
    }
}
