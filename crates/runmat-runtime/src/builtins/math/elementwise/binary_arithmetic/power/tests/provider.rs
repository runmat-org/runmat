use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn mixed_host_resident_power_preserves_low_rank_shape_owner_and_allocation() {
    test_support::with_test_provider(|provider| {
        for (shape, base_values, exponent_values, expected) in [
            (Vec::new(), vec![3.0], vec![2.0], vec![9.0]),
            (vec![2], vec![2.0, 3.0], vec![2.0, 3.0], vec![4.0, 27.0]),
        ] {
            let base = Tensor::new(base_values, shape.clone()).expect("resident base");
            let base = gpu_helpers::upload_tensor(provider, &base).expect("upload base");
            let exponent = Tensor::new(exponent_values, shape.clone()).expect("host exponent");
            let output = power_builtin(
                Value::GpuTensor(base.clone()),
                Value::Tensor(exponent),
                Vec::new(),
            )
            .expect("mixed host/resident power");
            let Value::GpuTensor(output) = output else {
                panic!("supported provider route must retain residency")
            };
            assert_eq!(output.shape(), shape);
            assert!(!output.shares_allocation_with(&base));
            assert!(runmat_accelerate_api::same_provider_instance(
                gpu_helpers::exact_provider_for_handle(&output).expect("output owner"),
                gpu_helpers::exact_provider_for_handle(&base).expect("input owner"),
            ));
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather output");
            assert_eq!(gathered.shape(), shape);
            assert_eq!(gathered.materialize_f64(), expected);
            base.release().expect("release base");
        }
    });
}

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
        base_handle.release().expect("release base");
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
        exponent_handle.release().expect("release exponent");
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
        for handle in [base_handle, invalid_handle, valid_handle] {
            let weak = handle.downgrade();
            handle.release().expect("release caller");
            assert!(weak.acquire().is_err());
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
        proto.release().expect("release prototype");
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
