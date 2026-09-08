#[cfg(feature = "wgpu")]
use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn arrayfun_wgpu_sin_matches_cpu() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };

    let tensor = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
    let result = call(
        Value::FunctionHandle("sin".into()),
        vec![Value::GpuTensor(handle.clone())],
    )
    .expect("arrayfun sin gpu");
    let Value::GpuTensor(out_handle) = result else {
        panic!("expected GPU tensor result");
    };
    let gathered = test_support::gather(Value::GpuTensor(out_handle.clone())).unwrap();
    let expected: Vec<f64> = values(&tensor).into_iter().map(f64::sin).collect();
    assert_eq!(gathered.shape, tensor.shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (actual, expect) in values(&gathered).iter().zip(expected.iter()) {
        assert!(
            (actual - expect).abs() < tol,
            "expected {expect}, got {actual}"
        );
    }
    let _ = provider.free(&handle);
    let _ = provider.free(&out_handle);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn arrayfun_wgpu_plus_matches_cpu() {
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
    let handle_a = gpu_helpers::upload_tensor(provider, &a).expect("upload a");
    let handle_b = gpu_helpers::upload_tensor(provider, &b).expect("upload b");
    let result = call(
        Value::FunctionHandle("plus".into()),
        vec![
            Value::GpuTensor(handle_a.clone()),
            Value::GpuTensor(handle_b.clone()),
        ],
    )
    .expect("arrayfun plus gpu");

    let Value::GpuTensor(out_handle) = result else {
        panic!("expected GPU tensor result");
    };
    let gathered = test_support::gather(Value::GpuTensor(out_handle.clone())).unwrap();
    let expected: Vec<f64> = values(&a)
        .iter()
        .zip(values(&b).iter())
        .map(|(x, y)| x + y)
        .collect();
    assert_eq!(gathered.shape, a.shape);
    let tol = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F64 => 1e-12,
        runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
    };
    for (actual, expect) in values(&gathered).iter().zip(expected.iter()) {
        assert!(
            (actual - expect).abs() < tol,
            "expected {expect}, got {actual}"
        );
    }
    let _ = provider.free(&handle_a);
    let _ = provider.free(&handle_b);
    let _ = provider.free(&out_handle);
}

#[test]
#[cfg(feature = "wgpu")]
fn arrayfun_wgpu_fallback_preserves_every_integer_class() {
    let _guard = test_support::accel_test_lock();
    if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .is_err()
    {
        return;
    }
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    for (storage, callback) in [
        (IntegerStorage::I8(vec![i8::MIN, i8::MAX]), "int8"),
        (IntegerStorage::I16(vec![i16::MIN, i16::MAX]), "int16"),
        (IntegerStorage::I32(vec![i32::MIN, i32::MAX]), "int32"),
        (IntegerStorage::I64(vec![i64::MIN, i64::MAX]), "int64"),
        (IntegerStorage::U8(vec![0, u8::MAX]), "uint8"),
        (IntegerStorage::U16(vec![0, u16::MAX]), "uint16"),
        (IntegerStorage::U32(vec![0, u32::MAX]), "uint32"),
        (
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            "uint64",
        ),
    ] {
        let handle = gpu_helpers::upload_tensor(
            provider,
            &Tensor::new_integer(storage.clone(), vec![1, 2]).expect("input"),
        )
        .expect("upload");
        let result = call(
            Value::FunctionHandle(callback.to_string()),
            vec![Value::GpuTensor(handle.clone())],
        )
        .expect("wgpu arrayfun");
        let Value::GpuTensor(output) = result else {
            panic!("expected resident output");
        };
        let gathered = test_support::gather(Value::GpuTensor(output.clone())).expect("gather");
        assert_eq!(gathered.integer_storage(), Some(&storage));
        let _ = provider.free(&handle);
        let _ = provider.free(&output);
    }

    let row = Tensor::new_integer(
        IntegerStorage::U64(vec![
            9_007_199_254_740_993,
            9_007_199_254_740_994,
            9_007_199_254_740_995,
        ]),
        vec![1, 3],
    )
    .expect("row");
    let row_handle = gpu_helpers::upload_tensor(provider, &row).expect("upload row");
    let column =
        Tensor::new_integer(IntegerStorage::U64(vec![10, 20]), vec![2, 1]).expect("column");
    let result = call(
        Value::FunctionHandle("plus".to_string()),
        vec![Value::GpuTensor(row_handle.clone()), Value::Tensor(column)],
    )
    .expect("compatible wgpu arrayfun");
    let Value::GpuTensor(output) = result else {
        panic!("expected resident output");
    };
    let gathered = test_support::gather(Value::GpuTensor(output.clone())).expect("gather");
    assert_eq!(gathered.shape, vec![2, 3]);
    assert_eq!(
        gathered.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            9_007_199_254_741_003,
            9_007_199_254_741_013,
            9_007_199_254_741_004,
            9_007_199_254_741_014,
            9_007_199_254_741_005,
            9_007_199_254_741_015,
        ]))
    );
    let _ = provider.free(&row_handle);
    let _ = provider.free(&output);
}
