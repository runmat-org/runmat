use super::*;

#[test]
fn unwrap_resident_controls_preserve_owner_cause_and_never_replay_downloads() {
    let phase =
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("host phase"));
    let empty_tolerance =
        Value::Tensor(runmat_value::Tensor::new(Vec::new(), vec![0, 0]).expect("empty tolerance"));
    for dimension in [false, true] {
        for fault in ["download", "expired", "foreign", "shape"] {
            let provider = Provider::new(Mode::Good, Upload::Good);
            let _scope = ThreadProviderGuard::set(Some(provider));
            let foreign = test_support::TestAllocationOwner::new();
            let owner = if fault == "foreign" {
                &foreign
            } else {
                &provider.owner
            };
            let control = owner
                .allocate_zeroed(
                    if fault == "shape" {
                        vec![2, 1]
                    } else {
                        vec![1, 1]
                    },
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                )
                .expect("borrowed control");
            if fault == "expired" {
                control.probe.mark_unavailable();
            }
            if fault == "download" {
                provider.fail_download.store(true, Ordering::SeqCst);
            }
            let mut arguments = vec![phase.clone()];
            if dimension {
                arguments.push(empty_tolerance.clone());
            }
            arguments.push(Value::GpuTensor(control.handle.clone()));
            let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
                "unwrap", &arguments, 1,
            ))
            .expect_err("invalid or unavailable resident control");
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
            assert_eq!(
                provider.downloads.load(Ordering::SeqCst),
                usize::from(fault == "download")
            );
            if fault == "download" {
                let mut cause: &(dyn std::error::Error + 'static) = &error;
                while !cause.is::<std::io::Error>() {
                    cause = cause.source().expect("typed control download cause");
                }
                assert_eq!(
                    cause.downcast_ref::<std::io::Error>().unwrap().kind(),
                    std::io::ErrorKind::BrokenPipe
                );
            }
            assert_eq!(control.probe.retirements(), 0);
        }
    }
}

#[test]
fn unwrap_successful_resident_controls_leave_borrowed_allocations_live() {
    let phase =
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![1, 2]).expect("host phase"));
    for dimension in [false, true] {
        let provider = Provider::new(Mode::Good, Upload::Good);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let control = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(vec![2.0]),
                shape: vec![1, 1],
                storage: GpuTensorStorage::Real,
            })
            .expect("resident control");
        *provider.source.lock().unwrap() = Some(control.probe.clone());
        let mut arguments = vec![phase.clone()];
        if dimension {
            arguments.push(Value::Tensor(
                runmat_value::Tensor::new(Vec::new(), vec![0, 0]).expect("empty tolerance"),
            ));
        }
        arguments.push(Value::GpuTensor(control.handle.clone()));
        let output = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "unwrap", &arguments, 1,
        ))
        .expect("control download and host phase execution");
        let Value::Tensor(output) = output.value() else {
            panic!("unwrap must publish a host tensor");
        };
        assert_eq!(output.materialize_f64(), vec![0.0, 1.0]);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        assert_eq!(control.probe.retirements(), 0);
        control.handle.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn envelope_resident_scalar_control_download_failure_precedes_signal_hook() {
    let provider = Provider::new(Mode::Good, Upload::Good);
    let _scope = ThreadProviderGuard::set(Some(provider));
    let source = input(provider, NumericElementType::F64, false);
    let control = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    provider.fail_download.store(true, Ordering::SeqCst);
    let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "envelope",
        &[
            Value::GpuTensor(source.handle.clone()),
            Value::GpuTensor(control.handle.clone()),
        ],
        1,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<std::io::Error>() {
        cause = cause.source().expect("control download cause");
    }
    assert_eq!(
        cause.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::BrokenPipe
    );
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(source.probe.retirements(), 0);
    assert_eq!(control.probe.retirements(), 0);
}

fn args(name: &str, source: Value) -> Vec<Value> {
    if name == "periodogram" {
        vec![
            source,
            Value::Tensor(runmat_value::Tensor::new(vec![1.0; 4], vec![4, 1]).unwrap()),
            Value::Num(4.0),
        ]
    } else {
        vec![source, Value::Num(2.0), Value::Num(0.0), Value::Num(4.0)]
    }
}

#[test]
fn spectral_controls_require_all_live_owners_before_first_download() {
    for name in ["periodogram", "spectrogram"] {
        for dead in [false, true] {
            let provider = Provider::new(Mode::Good, Upload::Good);
            let _scope = ThreadProviderGuard::set(Some(provider));
            let source = input(provider, NumericElementType::F64, false);
            let window = provider
                .owner
                .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let unrelated = test_support::TestAllocationOwner::new();
            let frequency = if dead { &provider.owner } else { &unrelated }
                .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            if dead {
                frequency.probe.mark_unavailable();
            }
            let mut arguments = args(name, Value::GpuTensor(source.handle.clone()));
            arguments[1] = Value::GpuTensor(window.handle.clone());
            *arguments.last_mut().unwrap() = Value::GpuTensor(frequency.handle.clone());
            let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
                name, &arguments, 2,
            ))
            .unwrap_err();
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert_eq!(source.probe.retirements(), 0);
            assert_eq!(window.probe.retirements(), 0);
        }
    }
}

#[test]
fn spectral_control_download_failure_is_terminal_and_keeps_original_cause() {
    for name in ["periodogram", "spectrogram"] {
        for window in [false, true] {
            let provider = Provider::new(Mode::Good, Upload::Good);
            let _scope = ThreadProviderGuard::set(Some(provider));
            let source = input(provider, NumericElementType::F64, false);
            let control = provider
                .owner
                .allocate_zeroed(vec![4, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let mut arguments = args(name, Value::GpuTensor(source.handle.clone()));
            let index = if window { 1 } else { arguments.len() - 1 };
            arguments[index] = Value::GpuTensor(control.handle.clone());
            provider.fail_download.store(true, Ordering::SeqCst);
            let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
                name, &arguments, 2,
            ))
            .unwrap_err();
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            let mut cause: &(dyn std::error::Error + 'static) = &error;
            while !cause.is::<std::io::Error>() {
                cause = cause.source().expect("typed control download failure");
            }
            assert_eq!(
                cause.downcast_ref::<std::io::Error>().unwrap().kind(),
                std::io::ErrorKind::BrokenPipe
            );
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
            assert_eq!(source.probe.retirements(), 0);
            assert_eq!(control.probe.retirements(), 0);
        }
    }
}

#[test]
fn typed_complex_zero_and_tiny_imaginary_signals_keep_default_twosided_range() {
    for name in ["periodogram", "spectrogram"] {
        for imaginary in [0.0, 1.0e-15] {
            let provider = Provider::new(Mode::Unsupported, Upload::Good);
            let _scope = ThreadProviderGuard::set(Some(provider));
            let pairs = [1.0, 2.0, 4.0, 8.0]
                .into_iter()
                .map(|real| (real, imaginary))
                .collect::<Vec<_>>();
            let source = provider
                .owner
                .allocate_numeric(HostNumericTensorOwned {
                    data: HostNumericDataOwned::F64(
                        pairs.iter().flat_map(|&(re, im)| [re, im]).collect(),
                    ),
                    shape: vec![4, 1],
                    storage: GpuTensorStorage::ComplexInterleaved,
                })
                .unwrap();
            *provider.source.lock().unwrap() = Some(source.probe.clone());
            let host_args = args(
                name,
                Value::ComplexTensor(runmat_value::ComplexTensor::new(pairs, vec![4, 1]).unwrap()),
            );
            let expected = block_on(runmat_runtime::call_builtin_async_with_outputs(
                name, &host_args, 2,
            ))
            .unwrap();
            let resident_args = args(name, Value::GpuTensor(source.handle.clone()));
            let actual = block_on(runmat_runtime::call_builtin_async_with_outputs(
                name,
                &resident_args,
                2,
            ))
            .unwrap();
            let Value::OutputList(values) = actual.value() else {
                panic!("two spectral outputs")
            };
            let Value::OutputList(expected) = expected.value() else {
                panic!("host reference outputs")
            };
            assert_eq!(values.len(), expected.len());
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
            for (actual, expected) in values.iter().zip(expected.iter()) {
                let Value::GpuTensor(handle) = actual else {
                    panic!("restored spectral output")
                };
                assert_eq!(handle.physical().element_type(), NumericElementType::F64);
                assert_eq!(
                    handle.physical().storage(),
                    if matches!(expected, Value::ComplexTensor(_)) {
                        GpuTensorStorage::ComplexInterleaved
                    } else {
                        GpuTensorStorage::Real
                    }
                );
                let gathered =
                    block_on(runmat_runtime::dispatcher::gather_if_needed_async(actual)).unwrap();
                assert_eq!(
                    &gathered, expected,
                    "native class, shape, complex storage and numerical values must agree"
                );
            }
            let Value::GpuTensor(frequency) = &values[1] else {
                panic!("restored frequency")
            };
            assert_eq!(frequency.shape(), &[4, 1]);
            assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 3);
            assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
            drop(actual);
            provider.assert_retired();
            assert_eq!(source.probe.retirements(), 0);
        }
    }
}
