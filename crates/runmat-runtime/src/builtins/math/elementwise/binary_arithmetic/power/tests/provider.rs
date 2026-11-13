use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn portable_mixed_scalar_power_keeps_exact_single_class_and_provider_owner() {
    test_support::with_test_provider(|provider| {
        let cases = [
            (
                Tensor::from_f32(vec![-4.0, 9.0], vec![2]).unwrap(),
                Value::Num(0.5),
                true,
                None,
                Some(vec![(0.0_f32, 2.0_f32), (3.0, 0.0)]),
            ),
            (
                Tensor::from_f32(vec![2.0, 3.0], vec![2]).unwrap(),
                Value::Num(-4.0),
                false,
                Some(vec![16.0_f32, -64.0]),
                None,
            ),
            (
                Tensor::new(vec![-4.0, 9.0], vec![2]).unwrap(),
                Value::Tensor(Tensor::from_f32(vec![0.5], vec![1, 1]).unwrap()),
                true,
                None,
                Some(vec![(0.0_f32, 2.0_f32), (3.0, 0.0)]),
            ),
            (
                Tensor::new(vec![3.0, 2.0], vec![2]).unwrap(),
                Value::Tensor(Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap()),
                false,
                Some(vec![-0.0_f32, 0.0]),
                None,
            ),
        ];
        for (resident_tensor, host, resident_is_base, expected_real, expected_complex) in cases {
            let resident = gpu_helpers::upload_tensor(provider, &resident_tensor).unwrap();
            let downloads = provider.telemetry_snapshot().download_bytes;
            let output = if resident_is_base {
                power_builtin(Value::GpuTensor(resident.clone()), host, Vec::new())
            } else {
                power_builtin(host, Value::GpuTensor(resident.clone()), Vec::new())
            }
            .expect("portable mixed power");
            let Value::GpuTensor(output) = output else {
                panic!("portable typed route must remain provider resident")
            };
            assert_eq!(provider.telemetry_snapshot().download_bytes, downloads);
            assert_eq!(output.shape(), [1, 2]);
            assert_eq!(
                runmat_accelerate_api::handle_precision(&output),
                Some(runmat_accelerate_api::ProviderPrecision::F32)
            );
            assert!(!output.shares_allocation_with(&resident));
            assert!(runmat_accelerate_api::same_provider_instance(
                gpu_helpers::exact_provider_for_handle(&output).expect("published owner"),
                provider,
            ));
            resident.release().unwrap();
            assert!(output.allocation_lease().validate_live().is_ok());
            let gathered = block_on(crate::dispatcher::gather_if_needed_async(
                &Value::GpuTensor(output.clone()),
            ))
            .unwrap();
            if let Some(expected) = expected_complex {
                let Value::ComplexTensor(gathered) = gathered else {
                    panic!("fractional negative base must retain complex Single")
                };
                assert!(gathered.as_f32_slice().is_some());
                assert_eq!(gathered.shape(), [1, 2]);
                for (actual, expected) in gathered.materialize_f64().into_iter().zip(expected) {
                    assert!((actual.0 - f64::from(expected.0)).abs() < 3e-5);
                    assert!((actual.1 - f64::from(expected.1)).abs() < 3e-5);
                }
            } else {
                let Value::Tensor(gathered) = gathered else {
                    panic!("integral exponent must retain real Single")
                };
                let actual = gathered.as_f32_slice().expect("Single real payload");
                let expected = expected_real.expect("real oracle");
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
            output.release().unwrap();
        }
    });
}

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

mod roundtrip;
