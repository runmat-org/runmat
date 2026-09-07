#[cfg(feature = "wgpu")]
use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn ldivide_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let lhs = Tensor::new(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
    let cpu = ldivide_builtin(
        Value::Tensor(lhs.clone()),
        Value::Tensor(rhs.clone()),
        Vec::new(),
    )
    .unwrap();
    let view_l = HostTensorView {
        data: &lhs.materialize_f64(),
        shape: &lhs.shape,
    };
    let view_r = HostTensorView {
        data: &rhs.materialize_f64(),
        shape: &rhs.shape,
    };
    let ha = provider.upload(&view_l).unwrap();
    let hb = provider.upload(&view_r).unwrap();
    let gpu = ldivide_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb), Vec::new())
        .expect("resident ldivide");
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(t) => {
            assert_eq!(gathered.materialize_f64().len(), t.materialize_f64().len());
            let tol = match provider.precision() {
                runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
            };
            for (ga, ca) in gathered
                .materialize_f64()
                .iter()
                .zip(t.materialize_f64().iter())
            {
                assert!((ga - ca).abs() < tol);
            }
        }
        Value::Num(n) => assert_eq!(gathered.materialize_f64(), vec![n]),
        other => panic!("unexpected cpu result {other:?}"),
    }
}
