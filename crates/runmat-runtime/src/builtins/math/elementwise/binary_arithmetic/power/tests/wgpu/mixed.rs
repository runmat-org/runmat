use super::super::*;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; run in the F64 qualification gate"]
fn mixed_single_base_double_scalar_keeps_native_rounding_and_complex_geometry() {
    let _guard = test_support::accel_test_lock();
    let provider = test_support::wgpu_provider_if_available()
        .expect("F64 qualification gate requires an actual WGPU adapter");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    for (values, exponent, complex, expected) in [
        (vec![2.0_f32, 3.0], 2.0, false, vec![(4.0, 0.0), (9.0, 0.0)]),
        (vec![-4.0_f32, 9.0], 0.5, true, vec![(0.0, 2.0), (3.0, 0.0)]),
    ] {
        let input = Tensor::from_f32(values, vec![2]).expect("single resident base");
        let base = gpu_helpers::upload_tensor(provider, &input).expect("typed single upload");
        let output = power_builtin(
            Value::GpuTensor(base.clone()),
            Value::Num(exponent),
            Vec::new(),
        )
        .expect("mixed native power");
        let Value::GpuTensor(output) = output else {
            panic!("supported mixed path must retain provider ownership")
        };
        assert_eq!(output.shape(), [1, 2]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            if complex {
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            } else {
                runmat_accelerate_api::GpuTensorStorage::Real
            }
        );
        assert!(!output.shares_allocation_with(&base));
        assert!(runmat_accelerate_api::same_provider_instance(
            gpu_helpers::exact_provider_for_handle(&output).expect("output owner"),
            provider,
        ));
        base.release().expect("release original input");
        let gathered = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(output),
        ))
        .expect("output survives input retirement");
        if complex {
            let Value::ComplexTensor(gathered) = gathered else {
                panic!("negative base fractional power must be complex")
            };
            assert_eq!(gathered.shape(), [1, 2]);
            assert!(gathered.as_f32_slice().is_some());
            for (actual, expected) in gathered.materialize_f64().into_iter().zip(expected) {
                assert!((actual.0 - expected.0).abs() < 2.0e-5);
                assert!((actual.1 - expected.1).abs() < 2.0e-5);
            }
        } else {
            let Value::Tensor(gathered) = gathered else {
                panic!("integral exponent must remain real")
            };
            assert_eq!(gathered.shape(), [1, 2]);
            assert!(gathered.as_f32_slice().is_some());
            for (actual, expected) in gathered.materialize_f64().into_iter().zip(expected) {
                assert!((actual - expected.0).abs() < 2.0e-5);
            }
        }
    }
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; run in the F64 qualification gate"]
fn mixed_single_host_base_double_resident_exponent_preserves_single_output() {
    let _guard = test_support::accel_test_lock();
    let provider = test_support::wgpu_provider_if_available()
        .expect("F64 qualification gate requires an actual WGPU adapter");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let exponents = Tensor::new(vec![2.0, 3.0], vec![2]).unwrap();
    let resident = gpu_helpers::upload_tensor(provider, &exponents).expect("double exponent");
    let host_base = Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap();
    let result = power_builtin(
        Value::Tensor(host_base),
        Value::GpuTensor(resident.clone()),
        Vec::new(),
    )
    .expect("mixed host base");
    let Value::GpuTensor(result) = result else {
        panic!("mixed result must remain native")
    };
    assert_eq!(result.shape(), [1, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&result),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
    assert!(!result.shares_allocation_with(&resident));
    resident.release().unwrap();
    let gathered = block_on(crate::dispatcher::gather_if_needed_async(
        &Value::GpuTensor(result),
    ))
    .expect("gather after input release");
    let Value::Tensor(gathered) = gathered else {
        panic!("integral exponents must remain real")
    };
    assert_eq!(gathered.shape(), [1, 2]);
    assert_eq!(gathered.as_f32_slice(), Some(&[4.0, 8.0][..]));
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; run in the F64 qualification gate"]
fn mixed_single_negative_zero_base_keeps_odd_exponent_sign() {
    let _guard = test_support::accel_test_lock();
    let provider = test_support::wgpu_provider_if_available()
        .expect("F64 qualification gate requires an actual WGPU adapter");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let exponent = Tensor::new(vec![-3.0, 3.0], vec![2]).unwrap();
    let exponent = gpu_helpers::upload_tensor(provider, &exponent).unwrap();
    let base = Tensor::from_f32(vec![-0.0], vec![1, 1]).unwrap();
    let result = power_builtin(
        Value::Tensor(base),
        Value::GpuTensor(exponent.clone()),
        Vec::new(),
    )
    .expect("mixed signed-zero power");
    let Value::GpuTensor(result) = result else {
        panic!("mixed signed-zero output must remain native")
    };
    assert_eq!(result.shape(), [1, 2]);
    assert_eq!(
        runmat_accelerate_api::handle_precision(&result),
        Some(runmat_accelerate_api::ProviderPrecision::F32)
    );
    exponent.release().unwrap();
    let gathered = block_on(crate::dispatcher::gather_if_needed_async(
        &Value::GpuTensor(result),
    ))
    .expect("signed-zero result after source retirement");
    let Value::Tensor(gathered) = gathered else {
        panic!("signed zero retains real domain")
    };
    let values = gathered.as_f32_slice().expect("single result");
    assert_eq!(values[0], f32::NEG_INFINITY);
    assert_eq!(values[1].to_bits(), (-0.0_f32).to_bits());
}
