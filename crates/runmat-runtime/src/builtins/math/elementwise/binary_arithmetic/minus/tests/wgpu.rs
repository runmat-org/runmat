use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_wgpu_matches_cpu_elementwise() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
    let cpu = minus_host(Value::Tensor(t.clone()), Value::Tensor(t.clone())).unwrap();
    let h = gpu_helpers::upload_tensor(runmat_accelerate_api::provider().unwrap(), &t).unwrap();
    let gpu = block_on(minus_gpu_pair(h.clone(), h)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(ct) => {
            assert_eq!(gathered.shape, ct.shape);
            for (a, b) in gathered
                .as_f64_slice()
                .expect("gathered double")
                .iter()
                .zip(ct.as_f64_slice().expect("CPU double"))
            {
                assert!((a - b).abs() < EPS);
            }
        }
        other => panic!("unexpected shapes {other:?}"),
    }
}
