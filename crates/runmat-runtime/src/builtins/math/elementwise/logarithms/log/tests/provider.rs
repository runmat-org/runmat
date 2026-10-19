use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::{IntegerStorage, Tensor};

#[test]
fn provider_result_remains_resident_and_matches_host() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
        let output = call(Value::GpuTensor(handle)).expect("log");
        assert!(matches!(output, Value::GpuTensor(_)));
        let gathered = test_support::gather(output).unwrap();
        for (actual, expected) in gathered
            .materialize_f64()
            .iter()
            .zip(input.materialize_f64().iter().map(|value| value.ln()))
        {
            assert!((*actual - expected).abs() < 1e-12);
        }
    });
}

#[test]
fn provider_fallback_preserves_complex_and_checks_integer_exactness() {
    test_support::with_test_provider(|provider| {
        let negative = Tensor::new(vec![-1.0, -2.0], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &negative).unwrap();
        let Value::GpuTensor(output) = call(Value::GpuTensor(handle)).unwrap() else {
            panic!("expected resident output")
        };
        assert!(matches!(
            block_on(gpu_helpers::download_value_preserving_residency_async(
                provider, &output
            ))
            .unwrap(),
            Value::ComplexTensor(_)
        ));

        let wide =
            Tensor::new_integer(IntegerStorage::U64(vec![9_007_199_254_740_993]), vec![1, 1])
                .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &wide).unwrap();
        assert!(call(Value::GpuTensor(handle)).is_err());
    });
}

#[test]
fn provider_contract_rejects_alias_shape_and_storage_mismatch() {
    test_support::with_test_provider(|provider| {
        let input =
            gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap())
                .unwrap();
        let output =
            gpu_helpers::upload_tensor(provider, &Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap())
                .unwrap();
        assert!(super::super::super::provider::contract::valid(
            &output, &input, provider
        ));
        assert!(!super::super::super::provider::contract::valid(
            &input, &input, provider
        ));
        let wrong_shape = output
            .with_view(runmat_accelerate_api::GpuTensorView::contiguous(vec![1, 2]))
            .expect("valid differently shaped view");
        assert!(!super::super::super::provider::contract::valid(
            &wrong_shape,
            &input,
            provider
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
        assert!(!super::super::super::provider::contract::valid(
            &alias_view,
            &input,
            provider,
        ));
        assert!(!super::super::super::provider::contract::valid(
            &wrong_storage,
            &input,
            provider
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
fn log_wgpu_matches_cpu_elementwise() {
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
    let input = Tensor::new(vec![1.0, 2.0, 4.0], vec![3, 1]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
    let gathered = test_support::gather(call(Value::GpuTensor(handle)).unwrap()).unwrap();
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(input.materialize_f64().iter().map(|value| value.ln()))
    {
        assert!((*actual - expected).abs() < 1e-5);
    }
}
