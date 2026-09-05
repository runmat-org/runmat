use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn imag_wgpu_matches_cpu_zero() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let tensor = Tensor::new(vec![0.0, 1.0, -2.5, 4.0], vec![4, 1]).unwrap();
    let cpu = imag_real(Value::Tensor(tensor.clone())).unwrap();
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let h = runmat_accelerate_api::AccelProvider::upload(provider, &view).unwrap();
    let gpu = block_on(imag_gpu(h)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    let cpu_tensor = match cpu {
        Value::Tensor(t) => t,
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
        other => panic!("unexpected cpu value {other:?}"),
    };
    assert_eq!(gathered.shape, cpu_tensor.shape);
    assert_eq!(
        gathered.materialize_f64().len(),
        cpu_tensor.materialize_f64().len()
    );
    for (g, c) in gathered
        .materialize_f64()
        .iter()
        .zip(cpu_tensor.materialize_f64().iter())
    {
        assert!((g - c).abs() < 1e-12, "imag mismatch {} vs {}", g, c);
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn imag_wgpu_complex_matches_cpu() {
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).unwrap();
    let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
    let gpu = block_on(imag_gpu(handle)).unwrap();
    let Value::GpuTensor(out) = gpu else {
        panic!("expected gpu tensor");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out),
        runmat_accelerate_api::GpuTensorStorage::Real
    );
    let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
    assert_eq!(gathered.materialize_f64(), vec![2.0, 4.5]);
}
