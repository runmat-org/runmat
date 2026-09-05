use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn conj_wgpu_matches_cpu_for_real() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let tensor = Tensor::new(vec![1.0, -2.0, 3.5, 0.0], vec![4, 1]).unwrap();
    let cpu = conj_real(Value::Tensor(tensor.clone())).unwrap();
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.materialize_f64(),
        shape: &tensor.shape,
    };
    let handle = runmat_accelerate_api::provider()
        .unwrap()
        .upload(&view)
        .unwrap();
    let gpu = block_on(conj_gpu(handle)).unwrap();
    let gathered = test_support::gather(gpu).expect("gather");
    match (cpu, gathered) {
        (Value::Tensor(ct), gt) => {
            assert_eq!(ct.shape, gt.shape);
            assert_eq!(ct.materialize_f64(), gt.materialize_f64());
        }
        _ => panic!("unexpected shapes"),
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn conj_wgpu_preserves_wide_uint64_identity_handle() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![2, 1],
    )
    .unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
    let buffer_id = handle.buffer_id;
    let Value::GpuTensor(output) = block_on(conj_gpu(handle)).expect("conj") else {
        panic!("expected resident integer");
    };
    assert_eq!(output.buffer_id, buffer_id);
    assert_eq!(
        runmat_accelerate_api::handle_integer_type(&output),
        Some(runmat_accelerate_api::IntegerElementType::U64)
    );
    let gathered =
        block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output))).expect("gather");
    let Value::Tensor(gathered) = gathered else {
        panic!("expected integer tensor");
    };
    assert_eq!(gathered.integer_storage(), tensor.integer_storage());
}

#[cfg(feature = "wgpu")]
#[test]
fn conj_wgpu_complex_matches_cpu() {
    let _guard = test_support::accel_test_lock();
    if !register_wgpu_provider_available() {
        return;
    }
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).unwrap();
    let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
    let gpu = block_on(conj_gpu(handle)).unwrap();
    let Value::GpuTensor(out) = gpu else {
        panic!("expected gpu tensor");
    };
    assert_eq!(
        runmat_accelerate_api::handle_storage(&out),
        runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    );
    let gathered =
        block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(out))).expect("gather");
    let Value::ComplexTensor(ct) = gathered else {
        panic!("expected complex tensor");
    };
    assert_eq!(ct.materialize_f64(), vec![(1.0, -2.0), (-3.0, 4.0)]);
}
