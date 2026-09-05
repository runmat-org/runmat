use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_builtins::LOG1P_ERROR_INVALID_INPUT;
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn provider_result_remains_resident_and_matches_host() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![0.0, -0.25, 0.5, 2.0], vec![4, 1]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let output = call(Value::GpuTensor(handle)).expect("provider log1p");
        assert!(matches!(output, Value::GpuTensor(_)));
        let gathered = test_support::gather(output).expect("gather");
        for (actual, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(input.materialize_f64().iter().map(|value| value.ln_1p()))
        {
            assert!((*actual - expected).abs() < 1e-12);
        }
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
            LOG1P_ERROR_INVALID_INPUT.identifier
        );
    });
}

#[test]
fn negative_resident_input_uses_owner_preserving_complex_fallback() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![-2.0, -3.0], vec![2, 1]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let Value::GpuTensor(output) = call(Value::GpuTensor(handle)).expect("log1p") else {
            panic!("expected resident output")
        };
        let Value::ComplexTensor(output) = block_on(
            gpu_helpers::download_value_preserving_residency_async(provider, &output),
        )
        .expect("download") else {
            panic!("expected complex tensor")
        };
        assert_eq!(output.shape, vec![2, 1]);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn log1p_wgpu_matches_cpu_elementwise() {
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
    let input = Tensor::new(vec![0.0, -0.25, 0.25, 1.0], vec![4, 1]).expect("input");
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
    let output = call(Value::GpuTensor(handle)).expect("log1p");
    let gathered = test_support::gather(output).expect("gather");
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(input.materialize_f64().iter().map(|value| value.ln_1p()))
    {
        assert!((*actual - expected).abs() < 1e-5);
    }
}
