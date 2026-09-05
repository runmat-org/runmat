use super::*;

#[test]
fn hypot_integer_gpu_pair_gathers_exact_storage_before_floating_provider_hook() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let wide = 9_007_199_254_740_992_u64;
        let left = Tensor::new_integer(IntegerStorage::U64(vec![wide, 3]), vec![1, 2]).unwrap();
        let right = Tensor::new_integer(IntegerStorage::U64(vec![1, 4]), vec![1, 2]).unwrap();
        let left = gpu_helpers::upload_tensor(provider, &left).expect("upload left");
        let right = gpu_helpers::upload_tensor(provider, &right).expect("upload right");
        let left_type = runmat_accelerate_api::handle_integer_type(&left);
        let right_type = runmat_accelerate_api::handle_integer_type(&right);
        let output = hypot_builtin(
            Value::GpuTensor(left.clone()),
            Value::GpuTensor(right.clone()),
        )
        .expect("integer gpu hypot");
        assert!(runmat_accelerate_api::provider_for_handle(&left).is_some());
        assert!(runmat_accelerate_api::provider_for_handle(&right).is_some());
        assert_eq!(runmat_accelerate_api::handle_integer_type(&left), left_type);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&right),
            right_type
        );
        let output = test_support::gather(output).expect("gather restored hypot");
        assert_eq!(
            output.into_numeric_storage().unwrap(),
            NumericStorage::F64(vec![(wide as f64).hypot(1.0), 5.0])
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_gpu_provider_roundtrip() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![3.0, 5.0, 8.0, 7.0], vec![2, 2]).unwrap();
        let rhs = Tensor::new(vec![4.0, 12.0, 15.0, 24.0], vec![2, 2]).unwrap();
        let h_lhs = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            })
            .expect("upload lhs");
        let h_rhs = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &rhs.materialize_f64(),
                shape: &rhs.shape,
            })
            .expect("upload rhs");
        let result =
            hypot_builtin(Value::GpuTensor(h_lhs), Value::GpuTensor(h_rhs)).expect("gpu hypot");
        let gathered = test_support::gather(result).expect("gathered result");
        let expected = [5.0, 13.0, 17.0, 25.0];
        assert_eq!(gathered.shape, vec![2, 2]);
        for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((actual - expect).abs() < 1e-12);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_gpu_and_host_mix_falls_back() {
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            })
            .expect("upload");
        let result =
            hypot_builtin(Value::GpuTensor(handle), Value::Num(4.0)).expect("gpu + host hypot");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        let expected: Vec<f64> = lhs
            .materialize_f64()
            .iter()
            .map(|&x| x.hypot(4.0))
            .collect();
        for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((actual - expect).abs() < 1e-12);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_gpu_left_host_integer_right_fallback_reads_integer_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let lhs = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &lhs.materialize_f64(),
                shape: &lhs.shape,
            })
            .expect("upload");
        let rhs = Tensor::new_integer(IntegerStorage::I16(vec![4, 3]), vec![2, 1]).unwrap();

        let precision = runmat_accelerate_api::handle_precision(&handle);
        let result = hypot_builtin(Value::GpuTensor(handle.clone()), Value::Tensor(rhs))
            .expect("gpu + integer host hypot");
        assert!(runmat_accelerate_api::provider_for_handle(&handle).is_some());
        assert_eq!(runmat_accelerate_api::handle_precision(&handle), precision);
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        let expected = [5.0, 5.0];
        for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn hypot_host_integer_left_gpu_right_fallback_reads_integer_storage() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let rhs = Tensor::new(vec![4.0, 3.0], vec![2, 1]).unwrap();
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &rhs.materialize_f64(),
                shape: &rhs.shape,
            })
            .expect("upload");
        let lhs = Tensor::new_integer(IntegerStorage::I16(vec![3, 4]), vec![2, 1]).unwrap();

        let result = hypot_builtin(Value::Tensor(lhs), Value::GpuTensor(handle))
            .expect("integer host + gpu hypot");
        let gathered = test_support::gather(result).expect("gather");
        assert_eq!(gathered.shape, vec![2, 1]);
        let expected = [5.0, 5.0];
        for (actual, expect) in gathered.materialize_f64().iter().zip(expected.iter()) {
            assert!((actual - expect).abs() < 1e-12, "{actual} vs {expect}");
        }
    });
}
