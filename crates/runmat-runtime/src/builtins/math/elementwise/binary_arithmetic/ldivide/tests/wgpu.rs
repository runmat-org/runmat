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
        shape: lhs.shape(),
    };
    let view_r = HostTensorView {
        data: &rhs.materialize_f64(),
        shape: rhs.shape(),
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

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual WGPU adapter with native integer transfer"]
fn ldivide_explicit_integer_fallback_preserves_public_operand_order() {
    use runmat_accelerate_api::AccelProvider;

    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required WGPU adapter");
    let input = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).unwrap();
    let source = gpu_helpers::upload_tensor(provider, &input)
        .unwrap()
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    // Tensor, not Num, makes the ordinary scalar provider route decline before effects.
    let scalar = || Value::Tensor(Tensor::new(vec![8.0], vec![1, 1]).unwrap());
    for (left_resident, expected) in [(true, vec![4, 2]), (false, vec![0, 1])] {
        let before = provider.telemetry_snapshot().download_bytes;
        let (left, right) = if left_resident {
            (Value::GpuTensor(source.clone()), scalar())
        } else {
            (scalar(), Value::GpuTensor(source.clone()))
        };
        let Value::GpuTensor(output) = ldivide_builtin(left, right, Vec::new()).unwrap() else {
            panic!("public ldivide fallback must publish a resident result")
        };
        assert!(
            provider.telemetry_snapshot().download_bytes > before,
            "source was gathered for fallback"
        );
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(output.shape(), &[2, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::I32)
        );
        assert!(!output.shares_allocation_with(&source));
        let gathered = test_support::gather(Value::GpuTensor(output)).unwrap();
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::I32(expected))
        );
        source.allocation_lease().validate_live().unwrap();
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual WGPU adapter with native integer transfer"]
fn ldivide_explicit_integer_scalar_uses_native_resident_division_in_both_orders() {
    use runmat_accelerate_api::AccelProvider;

    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required WGPU adapter");
    let input = Tensor::new_integer(IntegerStorage::I32(vec![2, 4]), vec![2, 1]).unwrap();
    let source = gpu_helpers::upload_tensor(provider, &input)
        .unwrap()
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    for (resident_on_left, expected) in [(true, vec![4, 2]), (false, vec![0, 1])] {
        provider.reset_telemetry();
        let (left, right) = if resident_on_left {
            (Value::GpuTensor(source.clone()), Value::Num(8.0))
        } else {
            (Value::Num(8.0), Value::GpuTensor(source.clone()))
        };
        let Value::GpuTensor(output) = ldivide_builtin(left, right, Vec::new())
            .expect("native integer scalar division remains resident")
        else {
            panic!("explicit integer result must remain resident")
        };
        assert!(runmat_accelerate_api::handle_is_explicit(&output));
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::I32)
        );
        assert!(!output.shares_allocation_with(&source));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(provider.telemetry_snapshot().upload_bytes > 0);
        let gathered = test_support::gather(Value::GpuTensor(output)).unwrap();
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::I32(expected))
        );
        source.allocation_lease().validate_live().unwrap();
    }
}
