use super::super::provider::single_from_gpu;
use super::super::storage::single_tensor_to_host;
use super::*;
use runmat_accelerate_api::AccelProvider;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn single_wgpu_matches_cpu() {
    let _state = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let tensor = Tensor::new(vec![0.2, 1.8, -3.4, 7.25], vec![2, 2]).unwrap();
    let cpu = single_tensor_to_host(tensor.clone()).expect("cpu conversion");
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).expect("upload");
    let gpu_value = block_on(single_from_gpu(handle)).expect("gpu single");
    let gathered = test_support::gather(gpu_value).expect("gather");
    assert_eq!(gathered.shape, cpu.shape);
    assert_eq!(gathered.materialize_f64(), cpu.materialize_f64());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn single_wgpu_large_ones_all_ones() {
    let _state = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let m = 200_000usize;
    let tensor = Tensor::ones(vec![m, 1]);
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).expect("upload");
    let gpu_value = block_on(single_from_gpu(handle)).expect("gpu single");
    let gathered = test_support::gather(gpu_value).expect("gather");
    assert_eq!(gathered.shape, tensor.shape);
    let sum: f64 = gathered.materialize_f64().iter().copied().sum();
    assert!(
        (sum - (m as f64)).abs() < 1e-9,
        "sum expected {} got {}",
        m,
        sum
    );
    assert!(gathered.materialize_f64().iter().all(|&v| v == 1.0));
}
