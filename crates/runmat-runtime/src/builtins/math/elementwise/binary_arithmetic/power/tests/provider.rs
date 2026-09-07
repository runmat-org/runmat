use super::*;
#[test]
fn power_gpu_left_rejects_nonscalar_host_integer_rhs() {
    test_support::with_test_provider(|provider| {
        let base = Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let base_handle = gpu_helpers::upload_tensor(provider, &base).expect("upload base");

        let exponent = Tensor::new_integer(IntegerStorage::U16(vec![3, 2]), vec![1, 2]).unwrap();

        let error = power_builtin(
            Value::GpuTensor(base_handle.clone()),
            Value::Tensor(exponent),
            Vec::new(),
        )
        .expect_err("mixed nonscalar integer and double arrays must reject");
        assert!(error
            .message()
            .contains("integer arrays can only be combined with scalar double"));
        let _ = provider.free(&base_handle);
    });
}

#[test]
fn power_gpu_right_rejects_nonscalar_host_integer_lhs() {
    test_support::with_test_provider(|provider| {
        let exponent = Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap();
        let exponent_handle =
            gpu_helpers::upload_tensor(provider, &exponent).expect("upload exponent");

        let base = Tensor::new_integer(IntegerStorage::U16(vec![2, 3]), vec![1, 2]).unwrap();

        let error = power_builtin(
            Value::Tensor(base),
            Value::GpuTensor(exponent_handle.clone()),
            Vec::new(),
        )
        .expect_err("mixed nonscalar integer and double arrays must reject");
        assert!(error
            .message()
            .contains("integer arrays can only be combined with scalar double"));
        let _ = provider.free(&exponent_handle);
    });
}

#[test]
fn power_gpu_integer_pairs_validate_exponents_before_provider_pow() {
    test_support::with_test_provider(|provider| {
        let base = Tensor::new_integer(IntegerStorage::I32(vec![2, -2]), vec![2, 1]).unwrap();
        let invalid_exponent =
            Tensor::new_integer(IntegerStorage::I32(vec![2, -1]), vec![1, 2]).unwrap();
        let base_handle = gpu_helpers::upload_tensor(provider, &base).unwrap();
        let invalid_handle = gpu_helpers::upload_tensor(provider, &invalid_exponent).unwrap();
        let error = power_builtin(
            Value::GpuTensor(base_handle.clone()),
            Value::GpuTensor(invalid_handle.clone()),
            Vec::new(),
        )
        .expect_err("resident negative integer exponent");
        assert_eq!(error.identifier(), Some("RunMat:power:InvalidInput"));
        assert!(error.message().contains("nonnegative integer values"));

        let valid_exponent =
            Tensor::new_integer(IntegerStorage::I32(vec![3, 0]), vec![1, 2]).unwrap();
        let valid_handle = gpu_helpers::upload_tensor(provider, &valid_exponent).unwrap();
        let result = power_builtin(
            Value::GpuTensor(base_handle.clone()),
            Value::GpuTensor(valid_handle.clone()),
            Vec::new(),
        )
        .expect("resident valid integer exponent");
        assert_eq!(
            result,
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I32(vec![8, -8, 1, 1]), vec![2, 2]).unwrap()
            )
        );
        for handle in [&base_handle, &invalid_handle, &valid_handle] {
            let _ = provider.free(handle);
        }
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_like_gpu_residency() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let exp = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let proto_tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let proto = gpu_helpers::upload_tensor(provider, &proto_tensor).expect("upload");
        let result = power_builtin(
            Value::Tensor(base.clone()),
            Value::Tensor(exp.clone()),
            vec![Value::from("like"), Value::GpuTensor(proto.clone())],
        )
        .expect("power");
        match result {
            Value::GpuTensor(handle) => {
                let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                let expected = [1.0, 8.0, 81.0];
                for (got, exp) in gathered
                    .as_f64_slice()
                    .expect("double result")
                    .iter()
                    .zip(expected.iter())
                {
                    assert!((got - exp).abs() < 1e-9);
                }
            }
            other => panic!("expected gpu result, got {other:?}"),
        }
        let _ = provider.free(&proto);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn power_gpu_pair_roundtrip() {
    test_support::with_test_provider(|provider| {
        let base = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let exp = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3]).unwrap();
        let hb = gpu_helpers::upload_tensor(provider, &base).expect("upload");
        let he = gpu_helpers::upload_tensor(provider, &exp).expect("upload");
        let result =
            power_builtin(Value::GpuTensor(hb), Value::GpuTensor(he), Vec::new()).expect("power");
        let gathered = test_support::gather(result).expect("gather");
        let expected = [1.0, 8.0, 81.0];
        for (got, exp) in gathered
            .as_f64_slice()
            .expect("double result")
            .iter()
            .zip(expected.iter())
        {
            assert!((got - exp).abs() < 1e-9);
        }
    });
}
