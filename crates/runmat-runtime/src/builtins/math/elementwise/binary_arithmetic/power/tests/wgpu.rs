#[cfg(feature = "wgpu")]
use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn power_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let exp = Tensor::new(vec![2.0, 0.5, -1.0], vec![1, 3]).unwrap();
    let cpu = power_host(Value::Tensor(base.clone()), Value::Tensor(exp.clone())).unwrap();
    let hb = gpu_helpers::upload_tensor(provider, &base).unwrap();
    let he = gpu_helpers::upload_tensor(provider, &exp).unwrap();
    let gpu = block_on(power_gpu_pair(hb.clone(), he.clone())).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    let cpu_tensor = match cpu {
        Value::Tensor(t) => t,
        Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
        other => panic!("unexpected cpu result {other:?}"),
    };
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-9,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (a, b) in gathered
        .as_f64_slice()
        .expect("gathered double")
        .iter()
        .zip(cpu_tensor.as_f64_slice().expect("CPU double"))
    {
        assert!((a - b).abs() < tol);
    }
    let _ = provider.free(&hb);
    let _ = provider.free(&he);
}
