use super::*;

fn args(name: &str, source: &GpuTensorHandle) -> Vec<Value> {
    if name == "periodogram" {
        vec![
            Value::GpuTensor(source.clone()),
            Value::Tensor(runmat_value::Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap()),
            Value::Num(4.0),
        ]
    } else {
        vec![
            Value::GpuTensor(source.clone()),
            Value::Num(2.0),
            Value::Num(0.0),
            Value::Num(4.0),
        ]
    }
}

fn call(
    name: &str,
    args: &[Value],
) -> Result<runmat_runtime::execution::RootedValue, Box<runmat_runtime::RuntimeError>> {
    block_on(runmat_runtime::call_builtin_async_with_outputs(
        name,
        args,
        if name == "periodogram" { 2 } else { 4 },
    ))
    .map_err(Box::new)
}

#[test]
fn spectral_producer_pair_is_admitted_before_coordinate_uploads() {
    for name in ["periodogram", "spectrogram"] {
        for mode in [
            Mode::Shape,
            Mode::Class,
            Mode::Logical,
            Mode::Foreign,
            Mode::Dead,
            Mode::Alias,
            Mode::Siblings,
        ] {
            let provider = Provider::new(mode, Upload::Good);
            let _scope = ThreadProviderGuard::set(Some(provider));
            let source = input(provider, NumericElementType::F64, true);
            let error = call(name, &args(name, &source.handle)).unwrap_err();
            assert_eq!(
                error.gpu_gather_retry,
                GpuGatherRetry::Never,
                "{name}: {mode:?}"
            );
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
            assert_eq!(provider.raw_uploads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            provider.assert_retired();
            source.handle.allocation_lease().validate_live().unwrap();
            assert_eq!(source.probe.retirements(), 0);
        }
    }
}

#[test]
fn coordinates_require_real_live_owned_encoding_and_requested_shape() {
    for name in ["periodogram", "spectrogram"] {
        for upload in [
            Upload::Logical,
            Upload::Complex,
            Upload::Integer,
            Upload::Foreign,
            Upload::Dead,
        ] {
            let provider = Provider::new(Mode::Good, upload);
            let _scope = ThreadProviderGuard::set(Some(provider));
            let source = input(provider, NumericElementType::F64, false);
            let error = call(name, &args(name, &source.handle)).unwrap_err();
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
            assert_eq!(provider.raw_uploads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            provider.assert_retired();
            assert_eq!(source.probe.retirements(), 0);
        }
        for stage in 1..=if name == "periodogram" { 1 } else { 2 } {
            for failure in [false, true] {
                let provider = Provider::new(Mode::Good, Upload::Good);
                if failure {
                    provider.fail_stage.store(stage, Ordering::SeqCst);
                } else {
                    provider.malformed_stage.store(stage, Ordering::SeqCst);
                }
                let _scope = ThreadProviderGuard::set(Some(provider));
                let source = input(provider, NumericElementType::F64, false);
                let error = call(name, &args(name, &source.handle)).unwrap_err();
                assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                if failure {
                    let mut cause: &(dyn std::error::Error + 'static) = &error;
                    while !cause.is::<std::io::Error>() {
                        cause = cause.source().expect("typed coordinate upload failure");
                    }
                    assert_eq!(
                        cause.downcast_ref::<std::io::Error>().unwrap().kind(),
                        std::io::ErrorKind::BrokenPipe
                    );
                }
                assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
                assert_eq!(provider.uploads.load(Ordering::SeqCst), stage);
                assert_eq!(provider.raw_uploads.load(Ordering::SeqCst), 0);
                assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
                provider.assert_retired();
                assert_eq!(source.probe.retirements(), 0);
            }
        }
    }
}

#[test]
fn time_coordinate_cannot_alias_a_distinct_frequency_view() {
    let provider = Provider::new(Mode::Good, Upload::Previous);
    let _scope = ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![6, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = call("spectrogram", &args("spectrogram", &source.handle)).unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.raw_uploads.load(Ordering::SeqCst), 0);
    provider.assert_retired();
    assert_eq!(source.probe.retirements(), 0);
}

#[test]
fn resident_window_is_protected_from_spectral_and_coordinate_aliases() {
    for name in ["periodogram", "spectrogram"] {
        for coordinate in [false, true] {
            let provider = Provider::new(
                if coordinate {
                    Mode::Good
                } else {
                    Mode::ControlAlias
                },
                if coordinate {
                    Upload::Alias
                } else {
                    Upload::Good
                },
            );
            let _scope = ThreadProviderGuard::set(Some(provider));
            let source = input(provider, NumericElementType::F64, false);
            let window = provider
                .owner
                .allocate_numeric(HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(vec![1.0; 4]),
                    shape: vec![4, 1],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            *provider.source.lock().unwrap() = Some(window.probe.clone());
            *provider.borrowed.lock().unwrap() = Some(window.handle.clone());
            let mut arguments = args(name, &source.handle);
            arguments[1] = Value::GpuTensor(window.handle.clone());
            *arguments.last_mut().unwrap() = Value::Num(6.0);
            let error = call(name, &arguments).unwrap_err();
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
            assert_eq!(
                provider.uploads.load(Ordering::SeqCst),
                usize::from(coordinate)
            );
            assert_eq!(provider.raw_uploads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            provider.borrowed.lock().unwrap().take();
            provider.assert_retired();
            assert_eq!(source.probe.retirements(), 0);
            assert_eq!(window.probe.retirements(), 0);
        }
    }
}

#[test]
fn pending_spectral_estimate_retires_owned_pair_not_borrowed_controls() {
    for name in ["periodogram", "spectrogram"] {
        let provider = Provider::new(Mode::Good, Upload::Good);
        provider.pending_stage.store(0, Ordering::SeqCst);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let source = input(provider, NumericElementType::F64, true);
        let window = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![1.0; 4]),
                shape: vec![4, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        *provider.source.lock().unwrap() = Some(window.probe.clone());
        let mut arguments = args(name, &source.handle);
        arguments[1] = Value::GpuTensor(window.handle.clone());
        let mut future = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
            name, &arguments, 2,
        ));
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&noop_waker()))
            .is_pending());
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        assert_eq!(provider.produced.lock().unwrap().len(), 2);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        drop(future);
        provider.assert_retired();
        assert_eq!(provider.raw_uploads.load(Ordering::SeqCst), 0);
        source.handle.allocation_lease().validate_live().unwrap();
        window.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(source.probe.retirements(), 0);
        assert_eq!(window.probe.retirements(), 0);
    }
}

#[test]
fn valid_native_spectral_outputs_preserve_caller_intent_and_signal_precision() {
    for name in ["periodogram", "spectrogram"] {
        for element in [NumericElementType::F32, NumericElementType::F64] {
            for explicit in [false, true] {
                let provider = Provider::new(Mode::Good, Upload::Good);
                let _scope = ThreadProviderGuard::set(Some(provider));
                let source = input(provider, element, explicit);
                let output = call(name, &args(name, &source.handle)).unwrap();
                let Value::OutputList(values) = output.value() else {
                    panic!("spectral output list")
                };
                for value in values {
                    let Value::GpuTensor(handle) = value else {
                        panic!("resident output")
                    };
                    assert_eq!(handle.physical().element_type(), element);
                    assert_eq!(
                        handle_provenance(handle),
                        Some(if explicit {
                            GpuHandleProvenance::Explicit
                        } else {
                            GpuHandleProvenance::Automatic
                        })
                    );
                    handle.allocation_lease().validate_live().unwrap();
                    assert!(!handle.shares_allocation_with(&source.handle));
                }
                assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
                assert_eq!(provider.raw_uploads.load(Ordering::SeqCst), 0);
                drop(output);
                provider.assert_retired();
                assert_eq!(source.probe.retirements(), 0);
            }
        }
    }
}
