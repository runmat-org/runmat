use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_value::Tensor;

#[test]
fn provider_result_remains_resident_and_matches_host() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![1.0, 10.0, 100.0], vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
        let output = call(Value::GpuTensor(handle)).unwrap();
        assert!(matches!(output, Value::GpuTensor(_)));
        let gathered = test_support::gather(output).unwrap();
        assert_eq!(gathered.materialize_f64(), &[0.0, 1.0, 2.0]);
    });
}

#[test]
fn negative_provider_input_uses_owner_preserving_complex_fallback() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new(vec![-10.0], vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
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
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn log10_wgpu_matches_cpu_elementwise() {
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
    let input = Tensor::new(vec![1.0, 10.0, 100.0], vec![3, 1]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &input).unwrap();
    let gathered = test_support::gather(call(Value::GpuTensor(handle)).unwrap()).unwrap();
    for (actual, expected) in gathered
        .materialize_f64()
        .iter()
        .zip(input.materialize_f64().iter().map(|value| value.log10()))
    {
        assert!((*actual - expected).abs() < 1e-5);
    }
}
