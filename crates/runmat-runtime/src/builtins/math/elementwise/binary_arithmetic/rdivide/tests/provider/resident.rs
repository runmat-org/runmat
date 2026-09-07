use super::super::*;
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn rdivide_gpu_pair_roundtrip() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![10.0, 20.0, 30.0], vec![3, 1]).unwrap();
        let rhs = Tensor::new(vec![2.0, 5.0, 10.0], vec![3, 1]).unwrap();
        let ha = gpu_helpers::upload_tensor(provider, &lhs).expect("upload lhs");
        let hb = gpu_helpers::upload_tensor(provider, &rhs).expect("upload rhs");
        let result = rdivide_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb), Vec::new())
            .expect("gpu rdivide");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![3, 1]);
        assert_eq!(double_values(&gathered), &[5.0, 4.0, 3.0]);
    });
}

#[test]
fn rdivide_complex_gpu_pair_stays_resident() {
    test_support::with_test_provider(|provider| {
        let lhs = ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap();
        let rhs = ComplexTensor::new(vec![(2.0, -1.0), (-1.0, 1.0)], vec![1, 2]).unwrap();
        let ha = gpu_helpers::upload_complex_tensor(provider, &lhs).expect("upload lhs");
        let hb = gpu_helpers::upload_complex_tensor(provider, &rhs).expect("upload rhs");

        let result = rdivide_builtin(Value::GpuTensor(ha), Value::GpuTensor(hb), Vec::new())
            .expect("gpu complex rdivide");
        let Value::GpuTensor(handle) = result else {
            panic!("expected resident gpu complex tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered =
            block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(handle))).unwrap();
        let Value::ComplexTensor(t) = gathered else {
            panic!("expected gathered complex tensor, got {gathered:?}");
        };
        assert_eq!(t.shape, vec![1, 2]);
        let expected = [(0.0, 1.0), (-3.5, 0.5)];
        for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
            assert!((got.0 - exp.0).abs() < EPS);
            assert!((got.1 - exp.1).abs() < EPS);
        }
    });
}

#[test]
fn rdivide_complex_gpu_scalar_paths_stay_resident() {
    test_support::with_test_provider(|provider| {
        let complex = ComplexTensor::new(vec![(1.0, 2.0), (-4.0, 0.5)], vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let div_result = rdivide_builtin(Value::GpuTensor(handle), Value::Num(2.0), Vec::new())
            .expect("gpu complex scalar divide");
        let Value::GpuTensor(div_handle) = div_result else {
            panic!("expected resident gpu complex tensor for scalar_div");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&div_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
            div_handle,
        )))
        .unwrap();
        let Value::ComplexTensor(t) = gathered else {
            panic!("expected gathered complex tensor, got {gathered:?}");
        };
        let expected = [(0.5, 1.0), (-2.0, 0.25)];
        for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
            assert!((got.0 - exp.0).abs() < EPS);
            assert!((got.1 - exp.1).abs() < EPS);
        }

        let handle = gpu_helpers::upload_complex_tensor(provider, &complex).expect("upload");
        let rdiv_result = rdivide_builtin(Value::Num(2.0), Value::GpuTensor(handle), Vec::new())
            .expect("gpu scalar complex divide");
        let Value::GpuTensor(rdiv_handle) = rdiv_result else {
            panic!("expected resident gpu complex tensor for scalar_rdiv");
        };
        assert_eq!(
            runmat_accelerate_api::handle_storage(&rdiv_handle),
            GpuTensorStorage::ComplexInterleaved
        );
        let gathered = block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(
            rdiv_handle,
        )))
        .unwrap();
        let Value::ComplexTensor(t) = gathered else {
            panic!("expected gathered complex tensor, got {gathered:?}");
        };
        let expected = [(0.4, -0.8), (-32.0 / 65.0, -4.0 / 65.0)];
        for (got, exp) in t.materialize_f64().iter().zip(expected.iter()) {
            assert!((got.0 - exp.0).abs() < EPS);
            assert!((got.1 - exp.1).abs() < EPS);
        }
    });
}
