use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn gpu_matrix_power_roundtrip() {
    test_support::with_test_provider(|provider| {
        let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &matrix).expect("upload");
        let result = mpower_builtin(Value::GpuTensor(handle), Value::Int(IntValue::I32(3)))
            .expect("gpu mpower");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 2]);
        assert_eq!(gathered.materialize_f64(), vec![37.0, 54.0, 81.0, 118.0]);
    });
}

#[test]
fn gpu_integer_matrix_power_roundtrips_exact_storage() {
    test_support::with_test_provider(|provider| {
        let matrix =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 0, 0, 1]), vec![2, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &matrix).expect("integer upload");
        let result =
            mpower_builtin(Value::GpuTensor(handle), Value::Num(1.0)).expect("gpu integer mpower");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(
            gathered.integer_storage(),
            Some(&IntegerStorage::U64(vec![u64::MAX, 0, 0, 1]))
        );
    });
}
