//! Actual F64 resident integration qualification.
use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires actual shader-f64 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn ordinary_resident_mixed_add_uses_three_recipes_and_declines_rank_change_to_host() {
    use runmat_accelerate_api::AccelProvider;
    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual F64 WGPU owner required");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let below = f32::from_bits(0x3f7f_ffff);
    let input = Tensor::from_f32(vec![below; 2], vec![2, 1]).unwrap();
    let resident = gpu_helpers::upload_tensor(provider, &input)
        .unwrap()
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let half = f64::from(f32::EPSILON) / 4.0;
    for scalar in [half - f64::EPSILON, half + f64::EPSILON] {
        for reverse in [false, true] {
            let operands = if reverse {
                [Value::Num(scalar), Value::GpuTensor(resident.clone())]
            } else {
                [Value::GpuTensor(resident.clone()), Value::Num(scalar)]
            };
            let [left, right] = operands;
            let Value::GpuTensor(output) = plus_builtin(left, right, Vec::new()).unwrap() else {
                panic!("admitted native F64 compound must remain resident");
            };
            assert!(runmat_accelerate_api::handle_is_explicit(&output));
            assert_eq!(
                output.physical().element_type(),
                runmat_accelerate_api::NumericElementType::F32
            );
            let gathered = test_support::gather(Value::GpuTensor(output)).unwrap();
            assert_eq!(gathered.as_f32_slice().unwrap().len(), 2);
            let expected = (f64::from(below) + scalar) as f32;
            assert!(gathered
                .as_f32_slice()
                .unwrap()
                .iter()
                .all(|value| value.to_bits() == expected.to_bits()));
        }
        let double = Value::Tensor(Tensor::new(vec![scalar], vec![1, 1, 1]).unwrap());
        let before = provider.telemetry_snapshot();
        let error = plus_builtin(Value::GpuTensor(resident.clone()), double, Vec::new())
            .expect_err("explicit rank-changing arithmetic requires a provider");
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ExplicitProviderRequired")
        );
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            before.download_bytes
        );
        resident.allocation_lease().validate_live().unwrap();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires actual shader-f64 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn ordinary_double_resident_single_scalar_both_orders_rank_and_provenance() {
    use runmat_accelerate_api::AccelProvider;
    let _guard = test_support::accel_test_lock();
    let provider = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual F64 owner");
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let half = f64::from(f32::EPSILON) / 2.0;
    let values = [1.0 + half - f64::EPSILON, 1.0 + half + f64::EPSILON];
    let source = Tensor::new(values.to_vec(), vec![2, 1]).unwrap();
    let base = gpu_helpers::upload_tensor(provider, &source).unwrap();
    for provenance in [
        runmat_accelerate_api::GpuHandleProvenance::Automatic,
        runmat_accelerate_api::GpuHandleProvenance::Explicit,
    ] {
        let resident = base.clone().with_provenance(provenance).unwrap();
        for control in [0.0_f32, -0.0, 1.0, f32::from_bits(0x7fc0_1234)] {
            for reverse in [false, true] {
                let scalar = Value::Tensor(Tensor::from_f32(vec![control], vec![1, 1]).unwrap());
                let (left, right) = if reverse {
                    (scalar, Value::GpuTensor(resident.clone()))
                } else {
                    (Value::GpuTensor(resident.clone()), scalar)
                };
                let Value::GpuTensor(output) = plus_builtin(left, right, Vec::new()).unwrap()
                else {
                    panic!("admitted mixed direction must remain resident");
                };
                assert_eq!(
                    runmat_accelerate_api::handle_provenance(&output),
                    Some(provenance)
                );
                assert_eq!(output.shape(), resident.shape());
                assert_eq!(
                    output.physical().element_type(),
                    runmat_accelerate_api::NumericElementType::F32
                );
                let gathered = test_support::gather(Value::GpuTensor(output)).unwrap();
                assert_eq!(gathered.as_f32_slice().unwrap().len(), values.len());
                for (&source, &actual) in values.iter().zip(gathered.as_f32_slice().unwrap()) {
                    let expected = (source + f64::from(control)) as f32;
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    }
                }
            }
            let before = provider.telemetry_snapshot();
            for reverse in [false, true] {
                let rank_changing =
                    Value::Tensor(Tensor::from_f32(vec![control], vec![1, 1, 1]).unwrap());
                let (left, right) = if reverse {
                    (rank_changing, Value::GpuTensor(resident.clone()))
                } else {
                    (Value::GpuTensor(resident.clone()), rank_changing)
                };
                match provenance {
                    runmat_accelerate_api::GpuHandleProvenance::Automatic => {
                        let Value::Tensor(output) = plus_builtin(left, right, Vec::new()).unwrap()
                        else {
                            panic!("automatic rank change may decline before mixed effects");
                        };
                        assert_eq!(output.shape(), &[2, 1, 1]);
                        assert_eq!(output.numeric_dtype(), NumericDType::F32);
                    }
                    runmat_accelerate_api::GpuHandleProvenance::Explicit => {
                        let before_download = provider.telemetry_snapshot().download_bytes;
                        let error = plus_builtin(left, right, Vec::new())
                            .expect_err("explicit rank change may not gather and compute");
                        assert_eq!(
                            error.identifier(),
                            Some("RunMat:gpu:ExplicitProviderRequired")
                        );
                        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                        assert_eq!(
                            provider.telemetry_snapshot().download_bytes,
                            before_download
                        );
                    }
                }
                assert_eq!(
                    provider.telemetry_snapshot().fused_elementwise.count,
                    before.fused_elementwise.count
                );
            }
            resident.allocation_lease().validate_live().unwrap();
        }
    }
}
