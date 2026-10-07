use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let result = imag_builtin(Value::GpuTensor(handle)).expect("imag");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape(), vec![4, 1]);
        assert!(gathered.materialize_f64().iter().all(|v| *v == 0.0));
    });
}

#[test]
fn imag_class_is_immutable_and_contradictory_wire_class_rejects_before_dispatch() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let original = handle.wire_descriptor();
        let acquired = provider
            .acquire_live(&original)
            .expect("exact live acquisition");
        assert!(acquired.shares_allocation_with(&handle));
        drop(acquired);
        let mut contradictory = original.clone();
        contradictory.physical = runmat_accelerate_api::GpuPhysicalDescriptor::new(
            runmat_accelerate_api::NumericElementType::F32,
            original.physical.storage(),
            original.physical.element_capacity(),
            original.physical.payload_bytes(),
            original.physical.requested_bytes(),
        )
        .unwrap();
        provider.reset_telemetry();
        let error = provider
            .acquire_live(&contradictory)
            .expect_err("changed native class");
        assert_eq!(
            error,
            runmat_accelerate_api::allocation::AllocationAcquisitionError::PhysicalMismatch
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(handle.wire_descriptor(), original);
        let weak = handle.downgrade();
        let result = block_on(imag_gpu(handle.clone())).expect("original valid input");
        let gathered = test_support::gather(result).expect("gather imag");
        assert_eq!(gathered.materialize_f64(), vec![0.0]);
        handle.release().expect("release caller");
        assert!(weak.acquire().is_err());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn imag_complex_gpu_provider_stays_resident() {
    test_support::with_test_provider(|provider| {
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = imag_builtin(Value::GpuTensor(handle)).expect("imag");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::Real
        );
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape(), vec![2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![2.0, 4.5]);
    });
}
