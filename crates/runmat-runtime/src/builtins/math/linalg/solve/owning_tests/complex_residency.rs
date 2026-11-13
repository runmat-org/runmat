//! Required-adapter complex continuation matrix for real-only linalg hooks.

use super::*;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate_api::{
    handle_is_explicit, handle_precision, handle_provenance, handle_storage,
    same_provider_instance, GpuHandleProvenance, HostNumericDataOwned, ProviderPrecision,
};
use runmat_value::ComplexTensor;

#[test]
#[ignore = "requires provisioned WGPU adapter; run with --ignored"]
fn complex_single_and_double_preserve_owner_class_and_intent_after_host_continuation() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("required WGPU adapter for complex linalg continuation");
    let _selected = ThreadProviderGuard::set(Some(provider));
    for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
        let matrix = match precision {
            ProviderPrecision::F32 => ComplexTensor::from_f32(
                vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (4.0, 0.0)],
                vec![2, 2],
            ),
            ProviderPrecision::F64 => ComplexTensor::new(
                vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (4.0, 0.0)],
                vec![2, 2],
            ),
        }
        .expect("diagonal complex matrix");
        for intent in [
            GpuHandleProvenance::Automatic,
            GpuHandleProvenance::Explicit,
        ] {
            let uploaded =
                crate::builtins::common::gpu_helpers::upload_complex_tensor(provider, &matrix)
                    .expect("complex source upload");
            let source = uploaded
                .with_provenance(intent)
                .expect("admit source placement intent");
            drop(uploaded);
            assert_eq!(handle_provenance(&source), Some(intent));
            for (operation, expected_scalar) in [
                ("inv", None),
                ("pinv", None),
                ("norm", Some(4.0)),
                ("rcond", Some(5.0_f64.sqrt() / 4.0)),
            ] {
                let rooted_output = futures::executor::block_on(crate::call_builtin_async(
                    operation,
                    &[Value::GpuTensor(source.clone())],
                ))
                .unwrap_or_else(|error| panic!("{operation} {precision:?} {intent:?}: {error}"));
                let Value::GpuTensor(output) = &*rooted_output else {
                    panic!("{operation} {precision:?} {intent:?} lost resident output");
                };
                assert!(!source.shares_allocation_with(output));
                source.allocation_lease().validate_live().unwrap();
                output.allocation_lease().validate_live().unwrap();
                assert!(
                    crate::builtins::common::gpu_helpers::exact_provider_for_handle(output)
                        .is_some_and(|owner| same_provider_instance(owner, provider))
                );
                assert_eq!(handle_precision(output), Some(precision));
                assert_eq!(handle_provenance(output), Some(intent));
                assert_eq!(
                    handle_is_explicit(output),
                    intent == GpuHandleProvenance::Explicit
                );
                assert_eq!(
                    handle_storage(output),
                    if expected_scalar.is_some() {
                        GpuTensorStorage::Real
                    } else {
                        GpuTensorStorage::ComplexInterleaved
                    }
                );
                let host = futures::executor::block_on(provider.download_numeric(output))
                    .expect("read back exact output owner");
                let tolerance = if precision == ProviderPrecision::F32 {
                    2e-5
                } else {
                    1e-10
                };
                match (expected_scalar, host.data) {
                    (Some(expected), HostNumericDataOwned::F32(values))
                        if precision == ProviderPrecision::F32 =>
                    {
                        assert_eq!(host.shape, vec![1, 1]);
                        assert!((f64::from(values[0]) - expected).abs() < tolerance);
                    }
                    (Some(expected), HostNumericDataOwned::F64(values))
                        if precision == ProviderPrecision::F64 =>
                    {
                        assert_eq!(host.shape, vec![1, 1]);
                        assert!((values[0] - expected).abs() < tolerance);
                    }
                    (None, HostNumericDataOwned::F32(values))
                        if precision == ProviderPrecision::F32 =>
                    {
                        assert_eq!(host.shape, vec![2, 2]);
                        assert_complex_inverse(
                            &values.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                            tolerance,
                        );
                    }
                    (None, HostNumericDataOwned::F64(values))
                        if precision == ProviderPrecision::F64 =>
                    {
                        assert_eq!(host.shape, vec![2, 2]);
                        assert_complex_inverse(&values, tolerance);
                    }
                    other => panic!(
                        "{operation} {precision:?} returned wrong physical output: {other:?}"
                    ),
                }
            }
        }
    }
}

fn assert_complex_inverse(interleaved: &[f64], tolerance: f64) {
    assert_eq!(interleaved.len(), 8);
    for (actual, expected) in interleaved
        .iter()
        .zip([0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0])
    {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected}"
        );
    }
}
