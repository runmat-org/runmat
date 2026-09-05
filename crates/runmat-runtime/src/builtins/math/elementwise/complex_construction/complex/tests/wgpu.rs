use super::*;

#[cfg(feature = "wgpu")]
#[test]
fn complex_wgpu_binary_matches_cpu_and_stays_resident() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let real = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
    let imag = Tensor::new(vec![-1.0, 0.5, 4.0], vec![3, 1]).unwrap();
    let expected = complex_call(
        Value::Tensor(real.clone()),
        vec![Value::Tensor(imag.clone())],
    )
    .expect("cpu complex");
    let real_handle = gpu_helpers::upload_tensor(provider, &real).expect("upload real");
    let imag_handle = gpu_helpers::upload_tensor(provider, &imag).expect("upload imag");
    let result = complex_call(
        Value::GpuTensor(real_handle),
        vec![Value::GpuTensor(imag_handle)],
    )
    .expect("gpu complex");
    let Value::GpuTensor(out) = result else {
        panic!("expected resident complex gpuArray");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );
    let gathered =
        block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
    assert_eq!(gathered, expected);
}

#[cfg(feature = "wgpu")]
#[test]
fn complex_wgpu_typed_uint64_composition_preserves_wide_components() {
    let _guard = test_support::accel_test_lock();
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        return;
    };
    let real = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![1, 2],
    )
    .expect("real");
    let imaginary =
        Tensor::new_integer(IntegerStorage::U64(vec![13, 17]), vec![1, 2]).expect("imaginary");
    let real = gpu_helpers::upload_tensor(provider, &real).expect("upload real");
    let imaginary = gpu_helpers::upload_tensor(provider, &imaginary).expect("upload imaginary");
    let Value::GpuTensor(output) =
        complex_call(Value::GpuTensor(real), vec![Value::GpuTensor(imaginary)]).expect("complex")
    else {
        panic!("expected resident complex integer");
    };
    let Value::ComplexTensor(gathered) =
        block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output))).expect("gather")
    else {
        panic!("expected complex integer");
    };
    let storage = gathered.integer_storage().expect("integer storage");
    assert_eq!(
        storage.real,
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX])
    );
    assert_eq!(storage.imag, IntegerStorage::U64(vec![13, 17]));
}

#[cfg(feature = "wgpu")]
#[test]
fn complex_wgpu_scalar_real_gpu_imag_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    ) else {
        return;
    };
    let imag = Tensor::new(vec![-1.0, 0.5, 4.0], vec![3, 1]).unwrap();
    let expected =
        complex_call(Value::Num(2.0), vec![Value::Tensor(imag.clone())]).expect("cpu complex");
    let imag_handle = gpu_helpers::upload_tensor(provider, &imag).expect("upload imag");
    let result =
        complex_call(Value::Num(2.0), vec![Value::GpuTensor(imag_handle)]).expect("gpu complex");
    let Value::GpuTensor(out) = result else {
        panic!("expected resident complex gpuArray");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );
    let gathered =
        block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
    assert_eq!(gathered, expected);
}
