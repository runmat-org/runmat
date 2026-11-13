use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mtimes_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let view_a = runmat_accelerate_api::HostTensorView {
            data: &a.materialize_f64(),
            shape: a.shape(),
        };
        let view_b = runmat_accelerate_api::HostTensorView {
            data: &b.materialize_f64(),
            shape: b.shape(),
        };
        let ha = provider.upload(&view_a).expect("upload A");
        let hb = provider.upload(&view_b).expect("upload B");
        let inputs = [
            ha.allocation_lease().downgrade(),
            hb.allocation_lease().downgrade(),
        ];
        let result = mtimes_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb)).expect("mtimes");
        let Value::GpuTensor(output) = &result else {
            panic!("resident mtimes output")
        };
        let output_lease = output.allocation_lease().downgrade();
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), [2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![26.0, 38.0, 30.0, 44.0]);
        assert!(inputs.iter().all(|lease| lease.acquire().is_none()));
        assert!(output_lease.acquire().is_none());
    });
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual WGPU adapter; run the explicit hardware gate"]
fn mtimes_wgpu_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("requested mtimes hardware gate requires an actual WGPU adapter");
    block_on(check_mtimes_wgpu_matches_cpu(provider));
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn mtimes_wgpu_matches_cpu() {
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider_async(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .await
    .expect("requested mtimes browser hardware gate requires WebGPU");
    check_mtimes_wgpu_matches_cpu(provider).await;
}

#[cfg(feature = "wgpu")]
async fn check_mtimes_wgpu_matches_cpu(
    provider: &'static runmat_accelerate::backend::wgpu::provider::WgpuProvider,
) {
    use runmat_accelerate_api::AccelProvider;
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

    let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

    let cpu = super::super::mtimes_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone()))
        .await
        .expect("cpu mtimes");
    let Value::Tensor(expected) = cpu else {
        panic!("host matrix product must be a tensor")
    };

    let view_a = runmat_accelerate_api::HostTensorView {
        data: &a.materialize_f64(),
        shape: a.shape(),
    };
    let view_b = runmat_accelerate_api::HostTensorView {
        data: &b.materialize_f64(),
        shape: b.shape(),
    };
    let ha = provider
        .upload(&view_a)
        .expect("upload A")
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("admit explicit input view");
    let hb = provider.upload(&view_b).expect("upload B");
    let inputs = [
        ha.allocation_lease().downgrade(),
        hb.allocation_lease().downgrade(),
    ];

    let gpu = super::super::mtimes_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb))
        .await
        .expect("wgpu mtimes");
    let Value::GpuTensor(output) = gpu else {
        panic!("explicit wgpu mtimes must remain resident");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert_eq!(output.device_id(), provider.device_id());
    let output_lease = output.allocation_lease().downgrade();
    let gathered = gpu_helpers::gather_tensor_async(&output)
        .await
        .expect("gather gpu");
    drop(output);

    assert_eq!(gathered.shape(), expected.shape());
    assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    assert!(inputs.iter().all(|lease| lease.acquire().is_none()));
    assert!(output_lease.acquire().is_none());
}
