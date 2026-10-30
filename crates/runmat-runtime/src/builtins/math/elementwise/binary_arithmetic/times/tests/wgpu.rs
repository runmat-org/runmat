use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual compatible WGPU adapter"]
fn times_wgpu_complex_single_equal_shape_and_broadcast_stay_resident() {
    use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};

    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required actual WGPU adapter");
    let complex = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, -4.0)], vec![2, 1]).unwrap();
    let paired = ComplexTensor::from_f32(vec![(2.0, -1.0), (-1.0, 1.0)], vec![2, 1]).unwrap();
    let real = Tensor::from_f32(vec![2.0, 0.5], vec![1, 2]).unwrap();
    for (rhs, shape, expected) in [
        (
            Value::GpuTensor(gpu_helpers::upload_complex_tensor(provider, &paired).unwrap()),
            vec![2, 1],
            vec![(4.0, 3.0), (1.0, 7.0)],
        ),
        (
            Value::GpuTensor(gpu_helpers::upload_tensor(provider, &real).unwrap()),
            vec![2, 2],
            vec![(2.0, 4.0), (6.0, -8.0), (0.5, 1.0), (1.5, -2.0)],
        ),
    ] {
        let left = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
        let result = times_builtin(Value::GpuTensor(left), rhs, Vec::new())
            .expect("complex WGPU multiplication");
        let handle = match result {
            Value::GpuTensor(handle) => handle,
            other => panic!("complex WGPU multiplication must remain resident, got {other:?}"),
        };
        assert_eq!(handle.shape(), shape);
        assert_eq!(
            handle.physical().storage(),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(handle.physical().element_type(), NumericElementType::F32);
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(handle),
        ))
        .expect("gather complex multiplication");
        let output = match gathered {
            Value::ComplexTensor(output) => output,
            other => panic!("expected complex output, got {other:?}"),
        };
        assert_eq!(output.shape(), shape);
        assert_eq!(output.materialize_f64(), expected);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn times_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
    let cpu = times_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
    let view_l = HostTensorView {
        data: &lhs.materialize_f64(),
        shape: lhs.shape(),
    };
    let view_r = HostTensorView {
        data: &rhs.materialize_f64(),
        shape: rhs.shape(),
    };
    let ha = provider.upload(&view_l).unwrap();
    let hb = provider.upload(&view_r).unwrap();
    let gpu = block_on(times_gpu_pair(None, ha, hb)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(t) => assert_eq!(gathered.materialize_f64(), t.materialize_f64()),
        Value::Num(n) => assert_eq!(gathered.materialize_f64(), vec![n]),
        other => panic!("unexpected cpu result {other:?}"),
    }
}
