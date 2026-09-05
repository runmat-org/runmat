use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn hypot_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let lhs = Tensor::new(vec![3.0, 4.0, 5.0, 12.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![4.0, 3.0, 12.0, 5.0], vec![2, 2]).unwrap();

    let cpu_value = compute_hypot_tensor(lhs.clone(), rhs.clone()).expect("cpu hypot");
    let expected = test_support::gather(cpu_value).expect("gather cpu result");

    let h_lhs = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &lhs.materialize_f64(),
            shape: &lhs.shape,
        })
        .expect("upload lhs");
    let h_rhs = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &rhs.materialize_f64(),
            shape: &rhs.shape,
        })
        .expect("upload rhs");

    let gpu_value =
        hypot_builtin(Value::GpuTensor(h_lhs), Value::GpuTensor(h_rhs)).expect("gpu hypot");
    let gathered = test_support::gather(gpu_value).expect("gather gpu result");

    assert_eq!(gathered.shape, expected.shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (actual, expect) in gathered
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
    {
        assert!(
            (actual - expect).abs() < tol,
            "|{actual} - {expect}| >= {tol}"
        );
    }
}
