use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
use runmat_builtins::{
    REALSQRT_ERROR_DOMAIN as ERROR_DOMAIN, REALSQRT_ERROR_INVALID_INPUT as ERROR_INVALID_INPUT,
};
use runmat_value::{ComplexTensor, Tensor};

#[test]
fn gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            })
            .unwrap();
        let result = call(Value::GpuTensor(handle)).unwrap();
        let gathered = test_support::gather(result).unwrap();
        assert_eq!(gathered.shape, vec![4, 1]);
        assert_eq!(gathered.materialize_f64(), vec![0.0, 1.0, 2.0, 3.0]);
    });
}

#[test]
fn gpu_negative_value_errors() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, -4.0], vec![1, 2]).unwrap();
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            })
            .unwrap();
        let err = call(Value::GpuTensor(handle)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_DOMAIN.identifier);
    });
}

#[test]
fn complex_gpu_input_errors_before_provider_sqrt() {
    test_support::with_test_provider(|provider| {
        let tensor = ComplexTensor::new(vec![(1.0, 0.0), (4.0, 2.0)], vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).unwrap();
        let err = call(Value::GpuTensor(handle)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    });
}

#[test]
fn integer_gpu_input_errors_before_provider_sqrt() {
    test_support::with_test_provider(|provider| {
        let values = [4u64, u64::MAX];
        let shape = [1usize, 2usize];
        let handle = provider
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U64(&values),
                shape: &shape,
            })
            .expect("upload integer gpu tensor");
        let err = call(Value::GpuTensor(handle)).unwrap_err();
        assert_eq!(err.identifier(), ERROR_INVALID_INPUT.identifier);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn realsqrt_wgpu_matches_cpu_for_nonnegative_values() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let tensor = Tensor::new(vec![0.0, 1.0, 4.0, 9.0], vec![4, 1]).unwrap();
    let expected = tensor
        .materialize_f64()
        .iter()
        .map(|value| value.sqrt())
        .collect::<Vec<_>>();
    let handle = runmat_accelerate_api::provider()
        .expect("WGPU provider")
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        })
        .expect("upload");
    let output = block_on(super::super::provider::evaluate(handle)).expect("realsqrt");
    let gathered = test_support::gather(output).expect("gather");
    let tolerance = match runmat_accelerate_api::provider().unwrap().precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    assert_eq!(gathered.shape, tensor.shape);
    for (actual, expected) in gathered.materialize_f64().iter().zip(expected) {
        assert!((actual - expected).abs() < tolerance);
    }
}
