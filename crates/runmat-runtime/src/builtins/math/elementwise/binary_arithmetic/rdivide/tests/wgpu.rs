#[cfg(feature = "wgpu")]
use super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn rdivide_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let lhs = Tensor::new(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
    let cpu = rdivide_builtin(
        Value::Tensor(lhs.clone()),
        Value::Tensor(rhs.clone()),
        Vec::new(),
    )
    .unwrap();
    let ha = gpu_helpers::upload_tensor(provider, &lhs).unwrap();
    let hb = gpu_helpers::upload_tensor(provider, &rhs).unwrap();
    let gpu = rdivide_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb), Vec::new())
        .expect("resident rdivide");
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(t) => {
            assert_eq!(gathered.len(), t.len());
            for (ga, ca) in double_values(&gathered).iter().zip(double_values(&t)) {
                assert!((ga - ca).abs() < EPS);
            }
        }
        Value::Num(n) => assert_eq!(double_values(&gathered), &[n]),
        other => panic!("unexpected cpu result {other:?}"),
    }
}
