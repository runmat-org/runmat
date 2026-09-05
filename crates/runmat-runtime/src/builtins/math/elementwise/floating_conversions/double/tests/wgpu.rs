use super::*;
use runmat_accelerate_api::AccelProvider;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn double_wgpu_matches_cpu() {
    let _state = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };

    let tensor = Tensor::new(vec![1.0, 2.5, -3.75, 4.125], vec![2, 2]).unwrap();
    let cpu_value = double_builtin(Value::Tensor(tensor.clone()), Vec::new()).unwrap();

    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = provider.upload(&view).expect("upload");
    let gpu_value = double_builtin(Value::GpuTensor(handle), Vec::new()).unwrap();

    let gathered = test_support::gather(gpu_value.clone()).expect("gather");
    match cpu_value {
        Value::Tensor(ref ct) => {
            assert_eq!(gathered.shape, ct.shape);
            assert_eq!(gathered.materialize_f64(), ct.materialize_f64());
        }
        Value::Num(n) => {
            assert_eq!(gathered.materialize_f64(), vec![n]);
        }
        other => panic!("unexpected CPU reference value {other:?}"),
    }

    if provider.precision() == ProviderPrecision::F64 {
        assert!(
            matches!(gpu_value, Value::GpuTensor(_)),
            "expected GPU residency under f64 precision"
        );
    } else {
        assert!(
            !matches!(gpu_value, Value::GpuTensor(_)),
            "expected host fallback when f64 unsupported"
        );
    }
}
