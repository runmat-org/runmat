use super::*;

fn signal(provider: &Provider, single: bool, explicit: bool) -> TestAllocation {
    let mut allocation = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: if single {
                HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0])
            } else {
                HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0, 4.0])
            },
            shape: vec![4, 1],
            storage: GpuTensorStorage::Real,
        })
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

fn arguments(name: &str, input: &GpuTensorHandle) -> Vec<Value> {
    let mut args = vec![
        Value::GpuTensor(input.clone()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap()),
    ];
    if name == "spectrogram" {
        args.push(Value::Num(0.0));
    }
    args.extend([
        Value::Num(4.0),
        Value::Num(4.0),
        Value::String("twosided".into()),
    ]);
    args
}

fn outputs(name: &str) -> usize {
    if name == "periodogram" {
        2
    } else {
        4
    }
}

#[test]
fn surplus_slots_remain_host_double_zero_on_direct_and_fallback_routes() {
    for name in ["periodogram", "spectrogram"] {
        for mode in [Mode::Good, Mode::Unsupported] {
            for single in [false, true] {
                let provider = Provider::new(mode, Upload::Good);
                let _scope = ThreadProviderGuard::set(Some(provider));
                let source = signal(provider, single, true);
                let args = arguments(name, &source.handle);
                let result = block_on(runmat_runtime::call_builtin_async_with_outputs(
                    name,
                    &args,
                    outputs(name) + 2,
                ))
                .unwrap();
                let Value::OutputList(values) = result.value() else {
                    panic!("padded outputs")
                };
                assert_eq!(values.len(), outputs(name) + 2);
                for value in &values[outputs(name)..] {
                    assert!(matches!(value, Value::Num(number) if *number == 0.0));
                }
                assert_eq!(
                    provider.uploads.load(Ordering::SeqCst),
                    if matches!(mode, Mode::Good) {
                        if name == "periodogram" {
                            1
                        } else {
                            2
                        }
                    } else {
                        outputs(name)
                    }
                );
                drop(result);
                provider.assert_retired();
                assert_eq!(source.probe.retirements(), 0);
            }
        }
    }
}

#[test]
fn unsupported_spectral_hook_restores_all_native_outputs_and_nonzero_payloads() {
    for name in ["periodogram", "spectrogram"] {
        for single in [false, true] {
            for explicit in [false, true] {
                let provider = Provider::new(Mode::Unsupported, Upload::Good);
                let _scope = ThreadProviderGuard::set(Some(provider));
                let source = signal(provider, single, explicit);
                let args = arguments(name, &source.handle);
                let result = block_on(runmat_runtime::call_builtin_async_with_outputs(
                    name,
                    &args,
                    outputs(name),
                ))
                .unwrap();
                let Value::OutputList(values) = result.value() else {
                    panic!("all outputs")
                };
                for (index, value) in values.iter().enumerate() {
                    let Value::GpuTensor(handle) = value else {
                        panic!("restored output")
                    };
                    assert_eq!(
                        handle.physical().element_type(),
                        if single {
                            NumericElementType::F32
                        } else {
                            NumericElementType::F64
                        }
                    );
                    assert_eq!(handle_is_explicit(handle), explicit);
                    let complex = name == "spectrogram" && index == 0;
                    assert_eq!(
                        handle.physical().storage(),
                        if complex {
                            GpuTensorStorage::ComplexInterleaved
                        } else {
                            GpuTensorStorage::Real
                        }
                    );
                    assert!(!handle.shares_allocation_with(&source.handle));
                    for prior in &values[..index] {
                        let Value::GpuTensor(prior) = prior else {
                            unreachable!()
                        };
                        assert!(!handle.shares_allocation_with(prior));
                    }
                    let payload = provider
                        .produced
                        .lock()
                        .unwrap()
                        .iter()
                        .find_map(|probe| {
                            probe
                                .with_data(handle, |data| match data {
                                    HostNumericDataView::F32(values) => {
                                        values.iter().map(|value| *value as f64).collect::<Vec<_>>()
                                    }
                                    HostNumericDataView::F64(values) => values.to_vec(),
                                    _ => panic!("native floating outputs"),
                                })
                                .ok()
                        })
                        .expect("actual published payload");
                    let expected = match (name, index) {
                        ("periodogram", 0) | ("spectrogram", 3) => vec![6.25, 0.5, 0.25, 0.5],
                        (_, 1) => vec![0.0, 1.0, 2.0, 3.0],
                        ("spectrogram", 2) => vec![0.5],
                        ("spectrogram", 0) => vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0],
                        _ => unreachable!(),
                    };
                    assert_eq!(payload, expected);
                }
                assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
                assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
                assert_eq!(provider.uploads.load(Ordering::SeqCst), outputs(name));
                drop(result);
                provider.assert_retired();
                assert_eq!(source.probe.retirements(), 0);
            }
        }
    }
}

#[test]
fn restoration_unsupported_respects_explicit_intent_without_replaying_hook() {
    for name in ["periodogram", "spectrogram"] {
        for single in [false, true] {
            for explicit in [false, true] {
                let provider = Provider::new(Mode::Unsupported, Upload::Unsupported);
                let _scope = ThreadProviderGuard::set(Some(provider));
                let source = signal(provider, single, explicit);
                let result = block_on(runmat_runtime::call_builtin_async_with_outputs(
                    name,
                    &arguments(name, &source.handle),
                    outputs(name),
                ));
                if explicit {
                    assert_eq!(result.unwrap_err().gpu_gather_retry, GpuGatherRetry::Never);
                    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
                } else {
                    let result = result.unwrap();
                    let Value::OutputList(values) = result.value() else {
                        panic!("all outputs")
                    };
                    for value in values {
                        let dtype = match value {
                            Value::Tensor(value) => value.numeric_dtype(),
                            Value::ComplexTensor(value) => value.numeric_dtype(),
                            _ => panic!("host fallback"),
                        };
                        assert_eq!(
                            dtype,
                            if single {
                                NumericDType::F32
                            } else {
                                NumericDType::F64
                            }
                        );
                    }
                    assert_eq!(provider.uploads.load(Ordering::SeqCst), outputs(name));
                    drop(result);
                }
                assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
                assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
                assert_eq!(source.probe.retirements(), 0);
            }
        }
    }
}

#[test]
fn second_restoration_failure_retires_first_output_and_preserves_source_cause() {
    for name in ["periodogram", "spectrogram"] {
        let provider = Provider::new(Mode::Unsupported, Upload::SecondFailure);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let source = signal(provider, true, true);
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            name,
            &arguments(name, &source.handle),
            outputs(name),
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while !cause.is::<std::io::Error>() {
            cause = cause.source().expect("retained operational upload cause");
        }
        assert_eq!(
            cause.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::BrokenPipe
        );
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        provider.assert_retired();
        assert_eq!(source.probe.retirements(), 0);
    }
}

#[test]
fn cancelling_pending_fallback_download_keeps_signal_lease_and_performs_no_upload() {
    for name in ["periodogram", "spectrogram"] {
        let provider = Provider::new(Mode::Unsupported, Upload::Good);
        provider.pending_download.store(true, Ordering::SeqCst);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let source = signal(provider, true, true);
        let args = arguments(name, &source.handle);
        let mut future = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
            name,
            &args,
            outputs(name),
        ));
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker()))
            .is_pending());
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        drop(future);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        provider.assert_retired();
        source.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(source.probe.retirements(), 0);
    }
}
