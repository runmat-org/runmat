#![cfg(not(target_arch = "wasm32"))]
#[path = "signal_output_ownership/provider.rs"]
mod provider;
#[path = "signal_output_ownership/preflight.rs"]
mod preflight;
#[path = "signal_output_ownership/spectral_producer.rs"]
mod spectral_producer;
#[path = "signal_output_ownership/spectral_precision.rs"]
mod spectral_precision;
#[path = "signal_output_ownership/siblings.rs"]
mod siblings;
#[path = "signal_output_ownership/welch.rs"]
mod welch;

use futures::{executor::block_on, task::noop_waker};
use provider::{Mode, Provider, Upload};
use runmat_accelerate_api::{test_support::TestAllocation, *};
use runmat_runtime::GpuGatherRetry;
use runmat_value::{NumericDType, Value};
use std::{future::Future, sync::atomic::Ordering, task::Context};

fn input(provider: &Provider, element: NumericElementType, explicit: bool) -> TestAllocation {
    let mut allocation = provider
        .owner
        .allocate_zeroed(vec![4, 1], element, GpuTensorStorage::Real)
        .unwrap();
    allocation.handle = allocation
        .handle
        .with_provenance(if explicit {
            GpuHandleProvenance::Explicit
        } else {
            GpuHandleProvenance::Automatic
        })
        .unwrap();
    *provider.source.lock().unwrap() = Some(allocation.probe.clone());
    allocation
}

#[test]
fn analytic_outputs_reject_malformed_owned_results_without_fallback() {
    for name in ["hilbert", "envelope"] {
        for mode in [
            Mode::Shape,
            Mode::Class,
            Mode::Logical,
            Mode::Foreign,
            Mode::Dead,
            Mode::Alias,
        ] {
            let provider = Provider::new(mode, Upload::Good);
            let _scope = ThreadProviderGuard::set(Some(provider));
            let input = input(provider, NumericElementType::F64, true);
            let args = [Value::GpuTensor(input.handle.clone())];
            let error = block_on(runmat_runtime::call_builtin_async(name, &args)).unwrap_err();
            assert_eq!(
                error.gpu_gather_retry,
                GpuGatherRetry::Never,
                "{name} {mode:?}: {error}"
            );
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            provider.assert_retired();
            input.handle.allocation_lease().validate_live().unwrap();
            assert!(handle_is_explicit(&input.handle));
            assert_eq!(input.probe.retirements(), 0);
            drop(error);
            drop(args);
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[test]
fn envelope_outputs_cannot_share_physical_storage() {
    let provider = Provider::new(Mode::Siblings, Upload::Good);
    let _scope = ThreadProviderGuard::set(Some(provider));
    let input = input(provider, NumericElementType::F64, false);
    let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "envelope",
        &[Value::GpuTensor(input.handle.clone())],
        2,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    provider.assert_retired();
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn analytic_unknown_or_dead_input_never_calls_unrelated_provider() {
    for name in ["hilbert", "envelope"] {
        for dead in [false, true] {
            let provider = Provider::new(Mode::Good, Upload::Good);
            let _scope = ThreadProviderGuard::set(Some(provider));
            let unknown = test_support::TestAllocationOwner::new();
            let owner = if dead { &provider.owner } else { &unknown };
            let input = owner
                .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            if dead {
                input.probe.mark_unavailable();
            }
            let error = block_on(runmat_runtime::call_builtin_async(
                name,
                &[Value::GpuTensor(input.handle.clone())],
            ))
            .unwrap_err();
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert_eq!(input.probe.retirements(), 0);
            drop(error);
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[test]
fn analytic_outputs_preserve_native_class_and_explicit_or_automatic_intent() {
    for name in ["hilbert", "envelope"] {
        for element in [NumericElementType::F32, NumericElementType::F64] {
            for explicit in [false, true] {
                for mode in [Mode::Good, Mode::Unsupported] {
                    let provider = Provider::new(mode, Upload::Good);
                    let _scope = ThreadProviderGuard::set(Some(provider));
                    let input = input(provider, element, explicit);
                    let owned_output = block_on(runmat_runtime::call_builtin_async(
                        name,
                        &[Value::GpuTensor(input.handle.clone())],
                    ))
                    .unwrap();
                    let Value::GpuTensor(output) = owned_output.value() else {
                        panic!("{name}: resident result required")
                    };
                    assert_eq!(output.shape(), &[4, 1]);
                    assert_eq!(output.physical().element_type(), element);
                    assert_eq!(handle_is_explicit(output), explicit);
                    assert_eq!(
                        output.physical().storage(),
                        if name == "hilbert" {
                            GpuTensorStorage::ComplexInterleaved
                        } else {
                            GpuTensorStorage::Real
                        }
                    );
                    assert!(!output.shares_allocation_with(&input.handle));
                    assert_eq!(
                        provider.hooks.load(Ordering::SeqCst),
                        usize::from(!matches!(mode, Mode::Unsupported))
                    );
                    assert_eq!(
                        provider.downloads.load(Ordering::SeqCst),
                        usize::from(matches!(mode, Mode::Unsupported))
                    );
                    assert_eq!(
                        provider.uploads.load(Ordering::SeqCst),
                        usize::from(matches!(mode, Mode::Unsupported))
                    );
                    provider
                        .produced
                        .lock()
                        .unwrap()
                        .iter()
                        .find_map(|probe| {
                            probe
                                .with_data(output, |data| match data {
                                    HostNumericDataView::F32(values) => {
                                        assert!(values.iter().all(|v| *v == 0.0))
                                    }
                                    HostNumericDataView::F64(values) => {
                                        assert!(values.iter().all(|v| *v == 0.0))
                                    }
                                    _ => panic!("native floating payload"),
                                })
                                .ok()
                        })
                        .expect("actual output payload");
                    drop(owned_output);
                    provider.assert_retired();
                    assert_eq!(input.probe.retirements(), 0);
                }
            }
        }
    }
}

#[test]
fn analytic_fallback_respects_typed_upload_failure_and_explicit_intent() {
    for name in ["hilbert", "envelope"] {
        for explicit in [false, true] {
            for upload in [Upload::Unsupported, Upload::Failure, Upload::Alias] {
                let provider = Provider::new(Mode::Unsupported, upload);
                let _scope = ThreadProviderGuard::set(Some(provider));
                let input = input(provider, NumericElementType::F32, explicit);
                if matches!(upload, Upload::Alias) {
                    *provider.borrowed.lock().unwrap() = Some(input.handle.clone());
                }
                let result = block_on(runmat_runtime::call_builtin_async(
                    name,
                    &[Value::GpuTensor(input.handle.clone())],
                ));
                if matches!(upload, Upload::Unsupported) && !explicit {
                    let output = result.unwrap();
                    match output.value() {
                        Value::Tensor(tensor) if name == "envelope" => {
                            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
                            assert_eq!(tensor.shape(), &[4, 1]);
                        }
                        Value::ComplexTensor(tensor) if name == "hilbert" => {
                            assert_eq!(tensor.numeric_dtype(), NumericDType::F32);
                            assert_eq!(tensor.shape(), &[4, 1]);
                        }
                        value => panic!("{name}: incorrect host fallback {value:?}"),
                    }
                } else {
                    let error = result.unwrap_err();
                    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                    if matches!(upload, Upload::Failure) {
                        let mut cause: &(dyn std::error::Error + 'static) = &error;
                        while !cause.is::<std::io::Error>() {
                            cause = cause.source().expect("original upload cause");
                        }
                        assert_eq!(
                            cause.downcast_ref::<std::io::Error>().unwrap().kind(),
                            std::io::ErrorKind::BrokenPipe
                        );
                    }
                }
                assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
                assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
                assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
                provider.borrowed.lock().unwrap().take();
                input.handle.allocation_lease().validate_live().unwrap();
                assert_eq!(input.probe.retirements(), 0);
            }
        }
    }
}

#[test]
fn analytic_call_cancellation_retires_provider_temporaries_not_callers() {
    for name in ["hilbert", "envelope"] {
        let provider = Provider::new(Mode::Pending, Upload::Good);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let input = input(provider, NumericElementType::F64, true);
        let args = [Value::GpuTensor(input.handle.clone())];
        let unpolled = runmat_runtime::call_builtin_async(name, &args);
        drop(unpolled);
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
        let mut future = Box::pin(runmat_runtime::call_builtin_async(name, &args));
        let waker = noop_waker();
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        assert!(!provider.produced.lock().unwrap().is_empty());
        drop(future);
        provider.assert_retired();
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
        assert!(handle_is_explicit(&input.handle));
    }
}
