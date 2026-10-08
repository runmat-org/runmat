use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual compatible WGPU adapter"]
fn plus_wgpu_complex_rank_beyond_shader_abi_uses_legal_host_fallback() {
    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("required actual WGPU adapter");
    let mut complex_shape = vec![1; 129];
    let mut real_shape = complex_shape.clone();
    complex_shape[0] = 2;
    real_shape[1] = 2;
    let complex = ComplexTensor::from_f32(vec![(1.0, 2.0), (3.0, 4.0)], complex_shape)
        .expect("complex high-rank input");
    let real = Tensor::from_f32(vec![10.0, 20.0], real_shape).expect("real high-rank input");
    let complex_handle =
        gpu_helpers::upload_complex_tensor(provider, &complex).expect("complex upload");
    let real_handle = gpu_helpers::upload_tensor(provider, &real).expect("real upload");
    let result = plus_builtin(
        Value::GpuTensor(complex_handle.clone()),
        Value::GpuTensor(real_handle.clone()),
        Vec::new(),
    )
    .expect("unsupported shader rank must use the legal host route");
    let Value::ComplexTensor(result) = result else {
        panic!("rank-limited provider hook must yield a host complex tensor");
    };
    let mut expected_shape = vec![1; 129];
    expected_shape[0] = 2;
    expected_shape[1] = 2;
    assert_eq!(result.shape(), expected_shape);
    assert_eq!(
        result.materialize_f64(),
        vec![(11.0, 2.0), (13.0, 4.0), (21.0, 2.0), (23.0, 4.0)]
    );
    assert!(complex_handle.allocation_lease().validate_live().is_ok());
    assert!(real_handle.allocation_lease().validate_live().is_ok());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn plus_wgpu_complex_gpu_stays_resident() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("provider");
    let shape = [2, 1];
    let real = provider
        .upload(&HostTensorView {
            data: &[1.0, -2.0],
            shape: &shape,
        })
        .expect("upload real");
    let imag = provider
        .upload(&HostTensorView {
            data: &[0.5, 4.0],
            shape: &shape,
        })
        .expect("upload imag");
    let complex =
        block_on(provider.complex_from_real_imag(&real, &imag)).expect("complex_from_real_imag");
    let offset = provider
        .upload(&HostTensorView {
            data: &[3.0, 7.0],
            shape: &shape,
        })
        .expect("upload offset");

    let result = plus_builtin(
        Value::GpuTensor(complex),
        Value::GpuTensor(offset),
        Vec::new(),
    )
    .expect("plus complex gpu");
    let handle = match result {
        Value::GpuTensor(handle) => handle,
        other => panic!("expected resident GPU result, got {other:?}"),
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&handle),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );
    let gathered = block_on(crate::dispatcher::gather_if_needed_async(
        &Value::GpuTensor(handle),
    ))
    .expect("gather complex gpu");
    match gathered {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape(), vec![2, 1]);
            assert_eq!(ct.materialize_f64(), vec![(4.0, 0.5), (5.0, 4.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn plus_wgpu_complex_scalar_implicit_expansion_stays_resident() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("provider");
    let scalar_shape = [1, 1];
    let real = provider
        .upload(&HostTensorView {
            data: &[2.0],
            shape: &scalar_shape,
        })
        .expect("upload real");
    let imag = provider
        .upload(&HostTensorView {
            data: &[-3.0],
            shape: &scalar_shape,
        })
        .expect("upload imag");
    let complex_scalar =
        block_on(provider.complex_from_real_imag(&real, &imag)).expect("complex scalar");
    let vector_shape = [3, 1];
    let vector = provider
        .upload(&HostTensorView {
            data: &[10.0, 20.0, 30.0],
            shape: &vector_shape,
        })
        .expect("upload vector");

    let result = plus_builtin(
        Value::GpuTensor(complex_scalar),
        Value::GpuTensor(vector),
        Vec::new(),
    )
    .expect("plus implicit expansion");
    let handle = match result {
        Value::GpuTensor(handle) => handle,
        other => panic!("expected resident GPU result, got {other:?}"),
    };
    assert_eq!(handle.shape(), &[3, 1]);
    assert_eq!(
        runmat_accelerate_api::handle_storage(&handle),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );
    let gathered = block_on(crate::dispatcher::gather_if_needed_async(
        &Value::GpuTensor(handle),
    ))
    .expect("gather complex result");
    match gathered {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape(), vec![3, 1]);
            assert_eq!(
                ct.materialize_f64(),
                vec![(12.0, -3.0), (22.0, -3.0), (32.0, -3.0)]
            );
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn plus_wgpu_complex_gpu_host_complex_scalar_falls_back() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("provider");
    let shape = [2, 1];
    let real = provider
        .upload(&HostTensorView {
            data: &[1.0, -2.0],
            shape: &shape,
        })
        .expect("upload real");
    let imag = provider
        .upload(&HostTensorView {
            data: &[0.5, 4.0],
            shape: &shape,
        })
        .expect("upload imag");
    let complex =
        block_on(provider.complex_from_real_imag(&real, &imag)).expect("complex_from_real_imag");

    let result = plus_builtin(
        Value::GpuTensor(complex),
        Value::Complex(10.0, -1.0),
        Vec::new(),
    )
    .expect("plus host complex scalar");
    match result {
        Value::ComplexTensor(ct) => {
            assert_eq!(ct.shape(), vec![2, 1]);
            assert_eq!(ct.materialize_f64(), vec![(11.0, -0.5), (8.0, 3.0)]);
        }
        other => panic!("expected host complex fallback, got {other:?}"),
    }
}
