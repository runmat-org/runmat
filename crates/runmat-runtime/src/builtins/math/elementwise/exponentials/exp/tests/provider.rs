use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_builtins::EXP_ERROR_INVALID_INPUT;
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn resident_integer_checks_exactness_before_floating_conversion() {
    test_support::with_test_provider(|provider| {
        let input =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .expect("integer input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let error = call(Value::GpuTensor(handle)).expect_err("inexact integer");
        assert_eq!(error.identifier(), EXP_ERROR_INVALID_INPUT.identifier);
    });
}

#[test]
fn provider_result_remains_resident_and_matches_host() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).expect("input");
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
        let output = call(Value::GpuTensor(handle)).expect("provider exp");
        assert!(matches!(output, Value::GpuTensor(_)));
        let gathered = test_support::gather(output).expect("gather");
        for (actual, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(input.materialize_f64().iter().map(|value| value.exp()))
        {
            assert!((*actual - expected).abs() < 1e-12);
        }
    });
}

#[test]
fn provider_contract_rejects_alias_shape_and_storage_mismatch() {
    test_support::with_test_provider(|provider| {
        let input = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("input"),
        )
        .expect("upload input");
        let output = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new(vec![1.0, 2.0], vec![2, 1]).expect("output"),
        )
        .expect("upload output");
        let expected_precision = runmat_accelerate_api::handle_precision(&input);
        assert!(super::super::super::provider::valid_real(
            &output,
            &input,
            provider,
            expected_precision,
        ));
        assert!(!super::super::super::provider::valid_real(
            &input,
            &input,
            provider,
            expected_precision,
        ));
        let wrong_shape = output
            .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![1, 2]))
            .expect("valid differently shaped view");
        assert!(!super::super::super::provider::valid_real(
            &wrong_shape,
            &input,
            provider,
            expected_precision,
        ));
        let wrong_storage = gpu_helpers::upload_complex_tensor(
            provider,
            &runmat_value::ComplexTensor::new(vec![(1.0, 0.0), (2.0, 0.0)], vec![2, 1]).unwrap(),
        )
        .expect("genuine complex allocation");
        let alias_view = input
            .with_view(input.view().clone())
            .expect("equal-shape view");
        assert_ne!(alias_view.view_identity(), input.view_identity());
        assert!(alias_view.shares_allocation_with(&input));
        assert!(!super::super::super::provider::valid_real(
            &alias_view,
            &input,
            provider,
            expected_precision,
        ));
        assert!(!super::super::super::provider::valid_real(
            &wrong_storage,
            &input,
            provider,
            expected_precision,
        ));
        let input_lease = input.allocation_lease().downgrade();
        let output_lease = output.allocation_lease().downgrade();
        let complex_lease = wrong_storage.allocation_lease().downgrade();
        drop((alias_view, wrong_shape, wrong_storage, input, output));
        assert!(input_lease.acquire().is_none());
        assert!(output_lease.acquire().is_none());
        assert!(complex_lease.acquire().is_none());
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn exp_wgpu_matches_cpu_elementwise() {
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
    let input = Tensor::new(vec![0.0, 1.0, 2.0], vec![3, 1]).expect("input");
    let handle = gpu_helpers::upload_tensor(provider, &input).expect("upload");
    let output = call(Value::GpuTensor(handle)).expect("exp");
    let gathered = test_support::gather(output).expect("gather");
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(input.materialize_f64().iter().map(|value| value.exp()))
    {
        assert!((*actual - expected).abs() < 1e-5);
    }
}
