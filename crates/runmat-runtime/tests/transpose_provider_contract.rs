use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::*;
use runmat_runtime::{call_builtin_async, GpuGatherRetry};
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[path = "transpose_provider_contract/provider.rs"]
mod fixture;
use fixture::{Mode, Provider};

fn invoke(
    name: &str,
    input: &GpuTensorHandle,
) -> Result<runmat_runtime::execution::RootedValue, Box<runmat_runtime::RuntimeError>> {
    block_on(call_builtin_async(name, &[Value::GpuTensor(input.clone())])).map_err(Box::new)
}

fn real(provider: &Provider) -> GpuTensorHandle {
    provider
        .inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1., 2., 3., 4., 5., 6.]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

#[test]
fn real_native_classes_transpose_without_conjugation_or_extra_transfers() {
    for name in ["transpose", "ctranspose"] {
        for data in [
            HostNumericDataOwned::F32(vec![1., 2., 3., 4., 5., 6.]),
            HostNumericDataOwned::F64(vec![1., 2., 3., 4., 5., 6.]),
            HostNumericDataOwned::U64(vec![u64::MAX, 2, 3, 4, 5, (1u64 << 53) + 1]),
        ] {
            let provider = Provider::new(Mode::Normal);
            let _active = ThreadProviderGuard::set(Some(provider));
            let host = HostNumericTensorOwned {
                data,
                shape: vec![2, 3],
                storage: GpuTensorStorage::Real,
            };
            let input = provider
                .inner
                .upload_numeric_raw(&host.as_view())
                .unwrap()
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap();
            let output = invoke(name, &input).unwrap();
            let Value::GpuTensor(handle) = output.value() else {
                panic!("resident result")
            };
            assert_eq!(handle.shape(), [3, 2]);
            assert_eq!(
                handle.physical().element_type(),
                input.physical().element_type()
            );
            assert!(handle_is_explicit(handle));
            let actual = block_on(provider.inner.download_numeric(handle)).unwrap();
            match actual.data {
                HostNumericDataOwned::F32(values) => assert_eq!(values, [1., 3., 5., 2., 4., 6.]),
                HostNumericDataOwned::F64(values) => assert_eq!(values, [1., 3., 5., 2., 4., 6.]),
                HostNumericDataOwned::U64(values) => {
                    assert_eq!(values, [u64::MAX, 3, 5, 2, 4, (1u64 << 53) + 1])
                }
                other => panic!("wrong class: {other:?}"),
            }
            assert_eq!(provider.counts(), (1, 0, 0, 0));
            let output_weak = handle.allocation_lease().downgrade();
            drop(output);
            assert!(output_weak.acquire().is_none());
            input.allocation_lease().validate_live().unwrap();
            assert_eq!(
                block_on(provider.inner.download_numeric(&input))
                    .unwrap()
                    .shape,
                host.shape
            );
        }
    }
}

#[test]
fn rejected_outputs_and_operational_errors_never_gather_or_replay() {
    for name in ["transpose", "ctranspose"] {
        for mode in [
            Mode::Failure,
            Mode::Shape,
            Mode::Class,
            Mode::Complex,
            Mode::Logical,
            Mode::Foreign,
        ] {
            let provider = Provider::new(mode);
            let _active = ThreadProviderGuard::set(Some(provider));
            let input = real(provider);
            let error = invoke(name, &input).unwrap_err();
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            if matches!(mode, Mode::Failure) {
                let mut cause: &(dyn std::error::Error + 'static) =
                    error.source.as_deref().unwrap();
                while !cause.is::<std::io::Error>() {
                    cause = cause.source().expect("original typed source");
                }
                assert_eq!(
                    cause.downcast_ref::<std::io::Error>().unwrap().kind(),
                    std::io::ErrorKind::BrokenPipe
                );
            }
            assert_eq!(provider.counts(), (1, 0, 0, 0));
            assert!(provider
                .produced
                .lock()
                .unwrap()
                .iter()
                .all(|lease| lease.acquire().is_none()));
            assert_eq!(
                block_on(provider.inner.download(&input)).unwrap().data,
                [1., 2., 3., 4., 5., 6.]
            );
        }
    }
}

#[test]
fn unsupported_restoration_uses_original_owner_and_preserves_explicit_intent() {
    for name in ["transpose", "ctranspose"] {
        for explicit in [false, true] {
            for mode in [
                Mode::Unsupported,
                Mode::UploadUnsupported,
                Mode::UploadFailure,
                Mode::UploadAlias,
            ] {
                let provider = Provider::new(mode);
                let _active = ThreadProviderGuard::set(Some(provider));
                let input = real(provider)
                    .with_provenance(if explicit {
                        GpuHandleProvenance::Explicit
                    } else {
                        GpuHandleProvenance::Automatic
                    })
                    .unwrap();
                *provider.original.lock().unwrap() = Some(input.downgrade());
                let result = invoke(name, &input);
                match mode {
                    Mode::Unsupported => {
                        let result = result.unwrap();
                        let Value::GpuTensor(output) = result.value() else {
                            panic!("restored resident result")
                        };
                        assert_eq!(output.shape(), [3, 2]);
                        assert_eq!(handle_is_explicit(output), explicit);
                        assert_eq!(
                            block_on(provider.inner.download(output)).unwrap().data,
                            [1., 3., 5., 2., 4., 6.]
                        );
                    }
                    Mode::UploadUnsupported if !explicit => {
                        let result = result.unwrap();
                        let Value::Tensor(output) = result.value() else {
                            panic!("automatic host fallback")
                        };
                        assert_eq!(output.shape(), &[3, 2]);
                        assert_eq!(output.materialize_f64(), [1., 3., 5., 2., 4., 6.]);
                    }
                    _ => {
                        let error = result.unwrap_err();
                        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                        if matches!(mode, Mode::UploadFailure) {
                            let NumericUploadError::Provider(cause) = error
                                .source
                                .as_ref()
                                .unwrap()
                                .downcast_ref::<NumericUploadError>()
                                .unwrap()
                            else {
                                panic!("typed provider upload failure")
                            };
                            assert_eq!(
                                std::error::Error::source(cause)
                                    .and_then(|source| source.downcast_ref::<std::io::Error>())
                                    .unwrap()
                                    .kind(),
                                std::io::ErrorKind::BrokenPipe
                            );
                        }
                    }
                }
                assert_eq!(provider.counts(), (1, 0, 1, 1));
                assert_eq!(
                    block_on(provider.inner.download(&input)).unwrap().data,
                    [1., 2., 3., 4., 5., 6.]
                );
            }
        }
    }
}

#[test]
fn conjugation_is_real_and_retains_intermediates_until_pending_call_is_dropped() {
    use std::{
        future::Future,
        task::{Context, Poll},
    };
    for explicit in [false, true] {
        let provider = Provider::new(Mode::ConjUnavailable);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .inner
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[
                    1., 2., 3., -4., 5., 6., 7., -8., 9., 10., 11., -12.,
                ]),
                shape: &[2, 3],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .unwrap()
            .with_provenance(if explicit {
                GpuHandleProvenance::Explicit
            } else {
                GpuHandleProvenance::Automatic
            })
            .unwrap();
        let result = invoke("ctranspose", &input).unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("completed-intermediate adaptation remains resident");
        };
        assert_eq!(handle_is_explicit(output), explicit);
        assert_eq!(output.shape(), [3, 2]);
        assert!(
            matches!(block_on(provider.inner.download_numeric(output)).unwrap().data,
            HostNumericDataOwned::F32(values) if values == [1., -2., 5., -6., 9., -10., 3., 4., 7., 8., 11., 12.])
        );
        assert_eq!(
            provider.counts(),
            (1, 1, 1, 1),
            "conjugation adapts the completed transpose exactly once"
        );
        input.allocation_lease().validate_live().unwrap();
    }
    for mode in [
        Mode::Normal,
        Mode::ConjAlias,
        Mode::ConjOriginal,
        Mode::ConjUnavailable,
        Mode::ConjFailure,
        Mode::Pending,
    ] {
        let provider = Provider::new(mode);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .inner
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[
                    1., 2., 3., -4., 5., 6., 7., -8., 9., 10., 11., -12.,
                ]),
                shape: &[2, 3],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .unwrap();
        *provider.original.lock().unwrap() = Some(input.downgrade());
        let args = [Value::GpuTensor(input.clone())];
        if matches!(mode, Mode::Pending) {
            let mut future = Box::pin(call_builtin_async("ctranspose", &args));
            let waker = futures::task::noop_waker();
            assert!(matches!(
                future.as_mut().poll(&mut Context::from_waker(&waker)),
                Poll::Pending
            ));
            assert!(provider.produced.lock().unwrap()[0].acquire().is_some());
            drop(future);
        } else if matches!(
            mode,
            Mode::ConjAlias | Mode::ConjOriginal | Mode::ConjFailure
        ) {
            assert_eq!(
                invoke("ctranspose", &input).unwrap_err().gpu_gather_retry,
                GpuGatherRetry::Never
            );
        } else {
            let result = invoke("ctranspose", &input).unwrap();
            let Value::GpuTensor(output) = result.value() else {
                panic!("complex resident result")
            };
            assert_eq!(output.shape(), [3, 2]);
            let actual = block_on(provider.inner.download_numeric(output)).unwrap();
            assert!(
                matches!(actual.data, HostNumericDataOwned::F64(values) if values == [1., -2., 5., -6., 9., -10., 3., 4., 7., 8., 11., 12.])
            );
        }
        assert_eq!(
            provider.counts(),
            if matches!(mode, Mode::ConjUnavailable) {
                (1, 1, 1, 1)
            } else {
                (1, 1, 0, 0)
            }
        );
        assert!(provider
            .produced
            .lock()
            .unwrap()
            .iter()
            .all(|lease| lease.acquire().is_none()));
        input.allocation_lease().validate_live().unwrap();
        assert_eq!(
            block_on(provider.inner.download_numeric(&input))
                .unwrap()
                .shape,
            [2, 3]
        );
    }
}

#[test]
fn rank_one_fallback_and_rank_zero_unknown_owner_preserve_the_real_boundary() {
    for name in ["transpose", "ctranspose"] {
        let provider = Provider::new(Mode::Normal);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .inner
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[1., 2., 3.]),
                shape: &[3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let result = invoke(name, &input).unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("restored rank-one vector")
        };
        assert_eq!(output.shape(), [1, 3]);
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        assert_eq!(provider.counts(), (1, 0, 1, 1));
        let unknown = test_support::TestAllocationOwner::new()
            .allocate_zeroed(vec![], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        assert_eq!(
            invoke(name, &unknown.handle).unwrap_err().gpu_gather_retry,
            GpuGatherRetry::Never
        );
        assert_eq!(provider.counts(), (1, 0, 1, 1));
        drop(unknown.handle);
        assert_eq!(unknown.probe.retirements(), 1);
    }
}

#[test]
fn real_aliases_are_lawful_and_complex_rank_zero_is_actually_conjugated() {
    for name in ["transpose", "ctranspose"] {
        let provider = Provider::new(Mode::Identity);
        let _active = ThreadProviderGuard::set(Some(provider));
        // A symmetric real matrix permits this provider's identity optimization.
        let input = provider
            .inner
            .upload(&HostTensorView {
                data: &[1., 2., 2., 3.],
                shape: &[2, 2],
            })
            .unwrap();
        let result = invoke(name, &input).unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("resident view")
        };
        assert!(output.shares_allocation_with(&input));
        assert_eq!(provider.counts(), (1, 0, 0, 0));
        drop(result);
        assert_eq!(
            block_on(provider.inner.download(&input)).unwrap().data,
            [1., 2., 2., 3.]
        );
    }
    let provider = Provider::new(Mode::Normal);
    let _active = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[3., -4.]),
            shape: &[],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    let result = invoke("ctranspose", &input).unwrap();
    let Value::GpuTensor(output) = result.value() else {
        panic!("complex scalar result")
    };
    assert!(output.shape().is_empty());
    assert!(!output.shares_allocation_with(&input));
    assert!(
        matches!(block_on(provider.inner.download_numeric(output)).unwrap().data, HostNumericDataOwned::F64(values) if values == [3., 4.])
    );
    assert_eq!(provider.counts(), (0, 1, 0, 0));
    assert!(
        matches!(block_on(provider.inner.download_numeric(&input)).unwrap().data, HostNumericDataOwned::F64(values) if values == [3., -4.])
    );
}

#[test]
fn logical_transpose_retains_producer_class_without_conjugation() {
    for name in ["transpose", "ctranspose"] {
        let provider = Provider::new(Mode::Normal);
        let _active = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .inner
            .upload_logical_raw(&HostLogicalTensorView {
                data: &[1, 0, 1, 1, 0, 0],
                shape: &[2, 3],
            })
            .unwrap();
        assert!(handle_is_logical(&input));
        assert_eq!(input.physical().element_type(), NumericElementType::F64);
        let result = invoke(name, &input).unwrap();
        let Value::GpuTensor(output) = result.value() else {
            panic!("resident logical result")
        };
        assert!(handle_is_logical(output));
        assert_eq!(output.shape(), [3, 2]);
        assert_eq!(
            output.physical().element_type(),
            input.physical().element_type()
        );
        assert!(
            matches!(block_on(provider.inner.download_numeric(output)).unwrap().data, HostNumericDataOwned::F64(values) if values == [1., 1., 0., 0., 1., 0.])
        );
        assert_eq!(provider.counts(), (1, 0, 0, 0));
        drop(result);
        input.allocation_lease().validate_live().unwrap();
    }
}
