use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual compatible WGPU adapter"]
fn minus_wgpu_complex_single_broadcast_stays_resident_in_both_operand_orders() {
    use runmat_accelerate_api::{GpuTensorStorage, NumericElementType};

    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required actual WGPU adapter");
    let complex = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, -4.0)], vec![2, 1]).unwrap();
    let real = Tensor::from_f32(vec![0.5, 2.0], vec![1, 2]).unwrap();
    let expected = [
        (
            false,
            vec![(0.5, 2.0), (2.5, -4.0), (-1.0, 2.0), (1.0, -4.0)],
        ),
        (
            true,
            vec![(-0.5, -2.0), (-2.5, 4.0), (1.0, -2.0), (-1.0, 4.0)],
        ),
    ];
    for (real_first, expected) in expected {
        let complex_handle = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
        let real_handle = gpu_helpers::upload_tensor(provider, &real).unwrap();
        let result = if real_first {
            minus_builtin(
                Value::GpuTensor(real_handle),
                Value::GpuTensor(complex_handle),
                Vec::new(),
            )
        } else {
            minus_builtin(
                Value::GpuTensor(complex_handle),
                Value::GpuTensor(real_handle),
                Vec::new(),
            )
        }
        .expect("complex WGPU subtraction");
        let handle = match result {
            Value::GpuTensor(handle) => handle,
            other => panic!("complex WGPU subtraction must remain resident, got {other:?}"),
        };
        assert_eq!(handle.shape(), &[2, 2]);
        assert_eq!(
            handle.physical().storage(),
            GpuTensorStorage::ComplexInterleaved
        );
        assert_eq!(handle.physical().element_type(), NumericElementType::F32);
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(handle),
        ))
        .expect("gather complex subtraction");
        let output = match gathered {
            Value::ComplexTensor(output) => output,
            other => panic!("expected complex output, got {other:?}"),
        };
        assert_eq!(output.shape(), vec![2, 2]);
        assert_eq!(output.materialize_f64(), expected);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn minus_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    let Some(provider) = test_support::wgpu_provider_if_available() else {
        return;
    };
    let t = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
    let cpu = minus_host(Value::Tensor(t.clone()), Value::Tensor(t.clone())).unwrap();
    let h = gpu_helpers::upload_tensor(provider, &t).unwrap();
    let gpu = block_on(minus_gpu_pair(None, h.clone(), h)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(ct) => {
            assert_eq!(gathered.shape(), ct.shape());
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
