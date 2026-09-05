use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5, -4.25], vec![4, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = real_builtin(Value::GpuTensor(handle)).expect("real");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![4, 1]);
        assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn real_complex_gpu_provider_stays_resident() {
    test_support::with_test_provider(|provider| {
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, 4.5)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = real_builtin(Value::GpuTensor(handle)).expect("real");
        let Value::GpuTensor(out) = result else {
            panic!("expected gpu tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&out),
            runmat_accelerate_api::GpuTensorStorage::Real
        );
        let gathered = test_support::gather(Value::GpuTensor(out)).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        assert_eq!(gathered.materialize_f64(), vec![1.0, -3.0]);
    });
}

#[test]
fn real_resident_wide_integer_identity_preserves_class_and_owner() {
    test_support::with_test_provider(|provider| {
        let input = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![1, 2],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &input).expect("integer upload");
        let Value::GpuTensor(output) = real_builtin(Value::GpuTensor(handle)).expect("real") else {
            panic!("documented gpuArray path must remain resident");
        };
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(runmat_accelerate_api::IntegerElementType::U64)
        );
        let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
        assert_eq!(
            gathered.integer_storage(),
            Some(&runmat_value::IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                u64::MAX,
            ]))
        );
    });
}
