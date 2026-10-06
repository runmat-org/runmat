use super::*;

mod complex;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn plus_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
    let cpu = plus_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let ha = gpu_helpers::upload_tensor(provider, &lhs).unwrap();
    let hb = gpu_helpers::upload_tensor(provider, &rhs).unwrap();
    let gpu = block_on(plus_gpu_pair(ha, hb)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(t) => assert_eq!(
            gathered.as_f64_slice().expect("double GPU output"),
            t.as_f64_slice().expect("double CPU output")
        ),
        Value::Num(n) => {
            assert_eq!(gathered.as_f64_slice().expect("double output"), &[n])
        }
        other => panic!("unexpected cpu result {other:?}"),
    }
}
