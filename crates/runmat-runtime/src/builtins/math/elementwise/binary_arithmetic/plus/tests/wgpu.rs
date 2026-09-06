use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn plus_wgpu_matches_cpu_elementwise() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let lhs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let rhs = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]).unwrap();
    let cpu = plus_host(Value::Tensor(lhs.clone()), Value::Tensor(rhs.clone())).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let ha = gpu_helpers::upload_tensor(provider, &lhs).unwrap();
    let hb = gpu_helpers::upload_tensor(provider, &rhs).unwrap();
    let gpu = block_on(plus_gpu_pair(ha, hb)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match cpu {
        Value::Tensor(t) => assert_eq!(
            gathered.as_f64_slice().expect("double GPU output"),
            t.as_f64_slice().expect("double CPU output")
        ),
        Value::Num(n) => {
            assert_eq!(gathered.as_f64_slice().expect("double output"), &[n])
        }
        other => panic!("unexpected cpu result {other:?}"),
    }
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
            assert_eq!(ct.shape, vec![2, 1]);
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
    assert_eq!(handle.shape, vec![3, 1]);
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
            assert_eq!(ct.shape, vec![3, 1]);
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
            assert_eq!(ct.shape, vec![2, 1]);
            assert_eq!(ct.materialize_f64(), vec![(11.0, -0.5), (8.0, 3.0)]);
        }
        other => panic!("expected host complex fallback, got {other:?}"),
    }
}
