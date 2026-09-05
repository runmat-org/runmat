use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![-3.0, -0.5, 0.0, 2.5], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = sign_builtin(Value::GpuTensor(handle)).expect("sign");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![-1.0, -1.0, 0.0, 1.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_gpu_complex_input_stays_resident_when_provider_supports_sign() {
    test_support::with_test_provider(|provider| {
        let tensor =
            ComplexTensor::new(vec![(3.0, 4.0), (0.0, 0.0), (-1.0, 1.0)], vec![3, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
        let result = sign_builtin(Value::GpuTensor(handle)).expect("sign");
        let Value::GpuTensor(out_handle) = result else {
            panic!("expected resident gpu output");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out_handle),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
            out_handle,
        )))
        .expect("gather");
        let Value::ComplexTensor(actual) = gathered else {
            panic!("expected complex tensor");
        };
        let expected = match sign_complex_tensor(tensor).expect("cpu sign") {
            Value::ComplexTensor(tensor) => tensor,
            other => panic!("expected complex tensor, got {other:?}"),
        };
        assert_eq!(actual.shape, expected.shape);
        for (got, want) in actual
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64().iter())
        {
            assert_complex_close(*got, *want, 1e-12);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn sign_gpu_complex_interleaved_rejects_odd_buffer_length() {
    test_support::with_test_provider(|provider| {
        let raw = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 1];
        let view = runmat_accelerate_api::HostTensorView {
            data: &raw,
            shape: &shape,
        };
        let mut handle = provider.upload(&view).expect("upload");
        handle.descriptor.storage =
            Some(runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved);
        let err =
            sign_builtin(Value::GpuTensor(handle)).expect_err("odd complex buffer should reject");
        assert!(
            err.message().contains("sign: internal error"),
            "unexpected error: {err}"
        );
        assert!(
            err.message()
                .contains("complex-interleaved buffer has odd length")
                || err.message().contains("TensorShapeError")
                || err.message().contains("shape"),
            "unexpected error detail: {err}"
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn sign_wgpu_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    if runmat_accelerate_api::provider().is_none() {
        return;
    }
    let tensor = Tensor::new(vec![-3.0, 0.0, 4.0, f64::NAN], vec![2, 2]).unwrap();
    let cpu = sign_real(Value::Tensor(tensor.clone())).unwrap();
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = runmat_accelerate_api::provider()
        .unwrap()
        .upload(&view)
        .unwrap();
    let gpu = block_on(super::super::provider::evaluate(handle)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match (cpu, gathered) {
        (Value::Tensor(ct), gt) => {
            assert_eq!(gt.shape, ct.shape);
            for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                if a.is_nan() && b.is_nan() {
                    continue;
                }
                assert_eq!(a, b);
            }
        }
        _ => panic!("unexpected shapes"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn sign_wgpu_complex_matches_cpu_and_stays_resident() {
    let _guard = test_support::accel_test_lock();
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let finite_max = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => f64::MAX,
        runmat_accelerate_api::ProviderPrecision::F32 => f32::MAX as f64,
    };
    let tensor = ComplexTensor::new(
        vec![
            (3.0, 4.0),
            (0.0, 0.0),
            (-1.0, 1.0),
            (finite_max, finite_max / 2.0),
            (f64::INFINITY, 2.0),
            (f64::INFINITY, f64::NEG_INFINITY),
            (f64::NAN, 1.0),
        ],
        vec![7, 1],
    )
    .unwrap();
    let handle = gpu_helpers::upload_complex_tensor(provider, &tensor).expect("upload");
    let gpu = block_on(super::super::provider::evaluate(handle)).expect("gpu sign");
    let Value::GpuTensor(out_handle) = gpu else {
        panic!("expected resident gpu output");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out_handle),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );
    let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
        out_handle,
    )))
    .expect("gather");
    let Value::ComplexTensor(actual) = gathered else {
        panic!("expected complex tensor");
    };
    let expected = match sign_complex_tensor(tensor).expect("cpu sign") {
        Value::ComplexTensor(tensor) => tensor,
        other => panic!("expected complex tensor, got {other:?}"),
    };
    assert_eq!(actual.shape, expected.shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (got, want) in actual
        .materialize_f64()
        .iter()
        .zip(expected.materialize_f64().iter())
    {
        assert_complex_close(*got, *want, tol);
    }
}
