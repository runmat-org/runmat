use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, -1.0, 0.5, -0.5], vec![2, 2]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = angle_builtin(Value::GpuTensor(handle)).expect("angle");
        let gathered = test_support::gather(result).expect("gather");
        let expected: Vec<f64> = tensor
            .materialize_f64()
            .iter()
            .map(|&v| angle_scalar(v, 0.0))
            .collect();
        assert_eq!(gathered.shape, vec![2, 2]);
        for (actual, target) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((actual - target).abs() < 1e-12);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_complex_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let complex = ComplexTensor::new(
            vec![(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)],
            vec![2, 2],
        )
        .unwrap();
        let handle =
            gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload complex");
        let result = angle_builtin(Value::GpuTensor(handle)).expect("angle");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, complex.shape);
        for (actual, (re, im)) in gathered
            .materialize_f64()
            .iter()
            .zip(complex.materialize_f64().iter())
        {
            assert!((actual - im.atan2(*re)).abs() < 1e-12);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn angle_rejects_all_native_integer_gpu_classes() {
    test_support::with_test_provider(|provider| {
        for storage in all_integer_storages() {
            let class = storage.class_name();
            let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
            let handle =
                gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer tensor");
            let error =
                angle_builtin(Value::GpuTensor(handle)).expect_err("integer gpuArray must reject");
            assert_eq!(
                error.identifier(),
                ANGLE_ERROR_INVALID_INPUT.identifier,
                "{class} gpuArray"
            );
            assert!(error.message().contains("integer gpuArray"));
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn angle_wgpu_rejects_all_native_integer_classes_before_float_dispatch() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    for storage in all_integer_storages() {
        let class = storage.class_name();
        let tensor = Tensor::new_integer(storage, vec![1, 2]).expect("integer tensor");
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer tensor");
        let error =
            angle_builtin(Value::GpuTensor(handle)).expect_err("integer WGPU input rejects");
        assert_eq!(
            error.identifier(),
            ANGLE_ERROR_INVALID_INPUT.identifier,
            "{class} WGPU"
        );
        assert!(error.message().contains("integer gpuArray"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn angle_wgpu_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let tensor = Tensor::new(vec![1.0, -1.0, 0.5, -0.5], vec![2, 2]).unwrap();
    let cpu = angle_tensor(tensor.clone()).unwrap();
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
    match (Value::Tensor(cpu), gathered) {
        (Value::Tensor(ct), gt) => {
            assert_eq!(gt.shape, ct.shape);
            let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
            };
            for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                assert!((a - b).abs() < tol);
            }
        }
        _ => panic!("unexpected shapes"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn angle_wgpu_complex_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let complex = ComplexTensor::new(
        vec![(3.0, 4.0), (-2.0, 5.0), (-1.5, -0.5), (2.5, -6.0)],
        vec![2, 2],
    )
    .unwrap();
    let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
    let result = block_on(super::super::provider::evaluate(handle)).unwrap();
    let gathered = test_support::gather(result).expect("gather");
    assert_eq!(gathered.shape, complex.shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (actual, (re, im)) in gathered
        .materialize_f64()
        .iter()
        .zip(complex.materialize_f64().iter())
    {
        assert!((actual - im.atan2(*re)).abs() < tol);
    }
}
