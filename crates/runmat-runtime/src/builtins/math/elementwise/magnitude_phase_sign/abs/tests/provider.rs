use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![-2.0, -1.0, 0.0, 3.0], vec![4, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = abs_builtin(Value::GpuTensor(handle)).expect("abs");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![4, 1]);
        assert_eq!(gathered.materialize_f64(), vec![2.0, 1.0, 0.0, 3.0]);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_gpu_preserves_exact_integer_class_and_values() {
    test_support::with_test_provider(|provider| {
        let values = [i64::MIN, -9_007_199_254_740_993, 0, i64::MAX];
        let shape = [2usize, 2usize];
        let handle = provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data: runmat_accelerate_api::HostIntegerDataView::I64(&values),
                shape: &shape,
            })
            .expect("upload integer gpu tensor");
        let result = abs_builtin(Value::GpuTensor(handle)).expect("abs");
        let Value::GpuTensor(ref output_handle) = result else {
            panic!("expected resident integer gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(output_handle),
            Some(runmat_accelerate_api::IntegerElementType::I64)
        );
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::I64(vec![
                i64::MAX,
                9_007_199_254_740_993,
                0,
                i64::MAX,
            ]))
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn abs_complex_gpu_provider_stays_resident() {
    test_support::with_test_provider(|provider| {
        let complex = ComplexTensor::new(vec![(3.0, 4.0), (1.0, -1.0)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = abs_builtin(Value::GpuTensor(handle)).expect("abs");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::Real
        );
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert!((gathered.materialize_f64()[0] - 5.0).abs() < 1e-12);
        assert!((gathered.materialize_f64()[1] - (2f64).sqrt()).abs() < 1e-12);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn abs_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let tensor = Tensor::new(vec![-3.0, -1.0, 0.5, -0.25], vec![4, 1]).unwrap();
    let cpu = abs_real(Value::Tensor(tensor.clone())).unwrap();
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let h = runmat_accelerate_api::provider()
        .unwrap()
        .upload(&view)
        .unwrap();
    let gpu = block_on(super::super::provider::evaluate(h)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match (cpu, gathered) {
        (Value::Tensor(ct), gt) => {
            assert_eq!(gt.shape, ct.shape);
            let tol = match runmat_accelerate_api::provider().unwrap().precision() {
                runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
                runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
            };
            for (a, b) in gt.materialize_f64().iter().zip(ct.materialize_f64().iter()) {
                assert!((*a - *b).abs() < tol, "|{} - {}| >= {}", a, b, tol);
            }
        }
        _ => panic!("unexpected result shape"),
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn abs_wgpu_complex_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let complex = ComplexTensor::new(vec![(3.0, 4.0), (1.0, -1.0)], vec![2, 1]).unwrap();
    let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
    let gpu = block_on(super::super::provider::evaluate(handle)).unwrap();
    let Value::GpuTensor(out) = gpu else {
        panic!("expected gpu tensor");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out),
        runmat_accelerate_api::GpuTensorStorage::Real
    );
    let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    assert!((gathered.materialize_f64()[0] - 5.0).abs() < tol);
    assert!((gathered.materialize_f64()[1] - (2f64).sqrt()).abs() < tol);
}

#[cfg(feature = "wgpu")]
#[test]
fn abs_wgpu_complex_preserves_infinity_with_nan_lane() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let complex = ComplexTensor::new(
        vec![(f64::INFINITY, f64::NAN), (f64::NAN, f64::NEG_INFINITY)],
        vec![2, 1],
    )
    .unwrap();
    let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
    let gpu = block_on(super::super::provider::evaluate(handle)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    assert_eq!(gathered.shape, vec![2, 1]);
    assert!(
        gathered.materialize_f64()[0].is_infinite()
            && gathered.materialize_f64()[0].is_sign_positive()
    );
    assert!(
        gathered.materialize_f64()[1].is_infinite()
            && gathered.materialize_f64()[1].is_sign_positive()
    );
}
