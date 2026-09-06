use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_wgpu_matches_cpu_elementwise() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
    let cpu = times_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
    let view_l = HostTensorView {
        data: &lhs.materialize_f64(),
        shape: &lhs.shape,
    };
    let view_r = HostTensorView {
        data: &rhs.materialize_f64(),
        shape: &rhs.shape,
    };
    let ha = runmat_accelerate_api::provider()
        .unwrap()
        .upload(&view_l)
        .unwrap();
    let hb = runmat_accelerate_api::provider()
        .unwrap()
        .upload(&view_r)
        .unwrap();
    let gpu = block_on(times_gpu_pair(ha, hb)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(t) => assert_eq!(gathered.materialize_f64(), t.materialize_f64()),
        Value::Num(n) => assert_eq!(gathered.materialize_f64(), vec![n]),
        other => panic!("unexpected cpu result {other:?}"),
    }
}
