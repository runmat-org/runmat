use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mtimes_gpu_roundtrip() {
    test_support::with_test_provider(|provider| {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();
        let view_a = runmat_accelerate_api::HostTensorView {
            data: &a.materialize_f64(),
            shape: &a.shape,
        };
        let view_b = runmat_accelerate_api::HostTensorView {
            data: &b.materialize_f64(),
            shape: &b.shape,
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
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![26.0, 38.0, 30.0, 44.0]);
        assert!(inputs.iter().all(|lease| lease.acquire().is_none()));
        assert!(output_lease.acquire().is_none());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn mtimes_wgpu_matches_cpu() {
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );

    let a = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

    let cpu =
        mtimes_builtin(Value::Tensor(a.clone()), Value::Tensor(b.clone())).expect("cpu mtimes");
    let expected = test_support::gather(cpu).expect("gather cpu");

    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let view_a = runmat_accelerate_api::HostTensorView {
        data: &a.materialize_f64(),
        shape: &a.shape,
    };
    let view_b = runmat_accelerate_api::HostTensorView {
        data: &b.materialize_f64(),
        shape: &b.shape,
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

    let gpu = mtimes_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb)).expect("wgpu mtimes");
    let Value::GpuTensor(output) = &gpu else {
        panic!("explicit wgpu mtimes must remain resident");
    };
    assert!(runmat_accelerate_api::handle_is_explicit(output));
    let output_lease = output.allocation_lease().downgrade();
    let gathered = test_support::gather(gpu).expect("gather gpu");

    assert_eq!(gathered.shape, expected.shape);
    assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    assert!(inputs.iter().all(|lease| lease.acquire().is_none()));
    assert!(output_lease.acquire().is_none());
}
