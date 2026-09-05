use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![0.0, 1.0, -3.0, 4.0], vec![4, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![4, 1]);
        assert_eq!(gathered.materialize_f64(), tensor.materialize_f64());
    });
}

#[test]
fn conj_resident_integer_is_exact_identity() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new_integer(
            IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
            vec![2, 1],
        )
        .unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload");
        let buffer_id = handle.buffer_id;
        let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
        let Value::GpuTensor(output) = result else {
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
            panic!("integer tensor");
        };
        assert_eq!(gathered.integer_storage(), tensor.integer_storage());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn conj_complex_gpu_provider_stays_resident() {
    test_support::with_test_provider(|provider| {
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (-3.0, -4.0)], vec![2, 1]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let result = conj_builtin(Value::GpuTensor(handle)).expect("conj");
        let Value::GpuTensor(out) = result else {
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
        assert_eq!(ct.shape, vec![2, 1]);
        assert_eq!(ct.materialize_f64(), vec![(1.0, -2.0), (-3.0, 4.0)]);
    });
}

#[test]
fn conj_typed_complex_integer_gpu_stays_exact_and_resident() {
    test_support::with_test_provider(|provider| {
        let complex = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
                IntegerStorage::I64(vec![i64::MIN, 17]),
            )
            .expect("storage"),
            vec![2, 1],
        )
        .expect("complex");
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let Value::GpuTensor(output) = conj_builtin(Value::GpuTensor(handle)).expect("conj") else {
            panic!("expected resident complex integer");
        };
        let Value::ComplexTensor(gathered) =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output))).expect("gather")
        else {
            panic!("expected complex integer");
        };
        let storage = gathered.integer_storage().expect("integer storage");
        assert_eq!(storage.real, IntegerStorage::I64(vec![i64::MIN, i64::MAX]));
        assert_eq!(storage.imag, IntegerStorage::I64(vec![i64::MAX, -17]));
    });
}
