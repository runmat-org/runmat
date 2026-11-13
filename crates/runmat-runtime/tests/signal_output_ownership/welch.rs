use super::*;

#[test]
fn window_and_frequency_download_failures_retain_cause_without_replay() {
    for argument in [1, 3] {
        let provider = Provider::new(Mode::Good, Upload::Good);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let source = input(provider, NumericElementType::F64, false);
        let control = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        provider.fail_download.store(true, Ordering::SeqCst);
        let mut arguments = args(&source.handle);
        arguments[argument] = Value::GpuTensor(control.handle.clone());
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "pwelch", &arguments, 2,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while !cause.is::<std::io::Error>() {
            cause = cause.source().expect("typed control download cause");
        }
        assert_eq!(
            cause.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::BrokenPipe
        );
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
        assert_eq!(source.probe.retirements(), 0);
        assert_eq!(control.probe.retirements(), 0);
    }
}

#[test]
fn pending_window_download_cancellation_preserves_all_borrowed_operands() {
    let provider = Provider::new(Mode::Good, Upload::Good);
    let _scope = ThreadProviderGuard::set(Some(provider));
    let source = input(provider, NumericElementType::F64, true);
    let window = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut arguments = args(&source.handle);
    arguments[1] = Value::GpuTensor(window.handle.clone());
    provider.pending_download.store(true, Ordering::SeqCst);
    let mut future = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
        "pwelch", &arguments, 2,
    ));
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&noop_waker()))
        .is_pending());
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    drop(future);
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    source.handle.allocation_lease().validate_live().unwrap();
    window.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(source.probe.retirements(), 0);
    assert_eq!(window.probe.retirements(), 0);
}

#[test]
fn explicit_window_is_caller_intent_not_a_provider_sibling() {
    let provider = Provider::new(Mode::Good, Upload::Good);
    let _scope = ThreadProviderGuard::set(Some(provider));
    let source = input(provider, NumericElementType::F64, false);
    let mut window = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 1.0]),
            shape: vec![2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    window.handle = window
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    *provider.source.lock().unwrap() = Some(window.probe.clone());
    let mut arguments = args(&source.handle);
    arguments[1] = Value::GpuTensor(window.handle.clone());
    let output = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "pwelch", &arguments, 2,
    ))
    .unwrap();
    let Value::OutputList(values) = output.value() else {
        panic!("two Welch outputs")
    };
    for value in values.iter() {
        let Value::GpuTensor(handle) = value else {
            panic!("explicit caller output")
        };
        assert_eq!(
            handle_provenance(handle),
            Some(GpuHandleProvenance::Explicit)
        );
        assert!(!handle.shares_allocation_with(&window.handle));
    }
    drop(output);
    provider.assert_retired();
    assert_eq!(source.probe.retirements(), 0);
    assert_eq!(window.probe.retirements(), 0);
}

#[test]
fn all_resident_controls_are_admitted_before_any_control_download() {
    for dead in [false, true] {
        let provider = Provider::new(Mode::Good, Upload::Good);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let source = input(provider, NumericElementType::F64, false);
        let window = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let unrelated = test_support::TestAllocationOwner::new();
        let control = if dead { &provider.owner } else { &unrelated }
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        if dead {
            control.probe.mark_unavailable();
        }
        let mut arguments = args(&source.handle);
        arguments[1] = Value::GpuTensor(window.handle.clone());
        arguments[3] = Value::GpuTensor(control.handle.clone());
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "pwelch", &arguments, 2,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 0);
        assert_eq!(source.probe.retirements(), 0);
        assert_eq!(window.probe.retirements(), 0);
    }
}

fn args(input: &GpuTensorHandle) -> Vec<Value> {
    vec![
        Value::GpuTensor(input.clone()),
        Value::Num(2.0),
        Value::Num(0.0),
        Value::Num(4.0),
    ]
}

#[test]
fn welch_nonzero_psd_and_typed_complex_fallback_match_host() {
    for imaginary in [None, Some(0.0), Some(1.0e-15)] {
        let provider = Provider::new(Mode::Unsupported, Upload::Good);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let real = vec![1.0, 2.0, 4.0, 8.0];
        let (data, storage, host) = if let Some(im) = imaginary {
            let pairs = real.iter().map(|&re| (re, im)).collect::<Vec<_>>();
            (
                pairs.iter().flat_map(|&(re, im)| [re, im]).collect(),
                GpuTensorStorage::ComplexInterleaved,
                Value::ComplexTensor(runmat_value::ComplexTensor::new(pairs, vec![4, 1]).unwrap()),
            )
        } else {
            (
                real.clone(),
                GpuTensorStorage::Real,
                Value::Tensor(runmat_value::Tensor::new(real, vec![4, 1]).unwrap()),
            )
        };
        let source = provider
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(data),
                shape: vec![4, 1],
                storage,
            })
            .unwrap();
        *provider.source.lock().unwrap() = Some(source.probe.clone());
        let mut host_args = args(&source.handle);
        host_args[0] = host;
        let host = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "pwelch", &host_args, 2,
        ))
        .unwrap();
        let resident_output = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "pwelch",
            &args(&source.handle),
            2,
        ))
        .unwrap();
        let (Value::OutputList(host), Value::OutputList(resident)) =
            (host.value(), resident_output.value())
        else {
            panic!("two Welch outputs")
        };
        let rows = if imaginary.is_some() { 4 } else { 3 };
        let mut expected = vec![76.5, 42.5, 8.5];
        if imaginary.is_some() {
            expected.push(42.5);
        } else {
            expected[1] *= 2.0;
        }
        for (index, (host, resident)) in host.iter().zip(resident.iter()).enumerate() {
            let (Value::Tensor(host), Value::GpuTensor(resident)) = (host, resident) else {
                panic!("host and restored native outputs")
            };
            assert_eq!(resident.shape(), &[rows, 1]);
            assert_eq!(host.shape(), resident.shape());
            let payload = provider
                .produced
                .lock()
                .unwrap()
                .iter()
                .find_map(|probe| {
                    probe
                        .with_data(resident, |data| match data {
                            HostNumericDataView::F64(values) => values.to_vec(),
                            _ => panic!("Double output"),
                        })
                        .ok()
                })
                .expect("live owned restored output");
            let host_values = host.materialize_f64();
            for (actual, host) in payload.iter().zip(host_values.iter()) {
                assert!((actual - host).abs() < 1.0e-12);
            }
            if index == 0 {
                for (actual, expected) in payload.iter().zip(&expected) {
                    assert!((actual - expected / (4.0 * std::f64::consts::PI)).abs() < 1.0e-12);
                }
            }
        }
        assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        drop(resident_output);
        provider.assert_retired();
        assert_eq!(source.probe.retirements(), 0);
    }
}

#[test]
fn welch_spectral_pair_requires_real_geometry_class_owner_and_independence() {
    for mode in [
        Mode::Shape,
        Mode::Class,
        Mode::Foreign,
        Mode::Dead,
        Mode::Logical,
        Mode::Alias,
    ] {
        let provider = Provider::new(mode, Upload::Good);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let input = input(provider, NumericElementType::F64, true);
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "pwelch",
            &args(&input.handle),
            2,
        ))
        .unwrap_err();
        assert_eq!(
            error.gpu_gather_retry,
            GpuGatherRetry::Never,
            "{mode:?}: {error}"
        );
        assert_eq!(provider.stage.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
        provider.assert_retired();
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[test]
fn each_welch_continuation_admits_output_before_next_effect() {
    for stage in 0..=4 {
        let provider = Provider::new(Mode::Good, Upload::Good);
        provider.malformed_stage.store(stage, Ordering::SeqCst);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let input = input(provider, NumericElementType::F64, true);
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "pwelch",
            &args(&input.handle),
            2,
        ))
        .unwrap_err();
        assert_eq!(
            error.gpu_gather_retry,
            GpuGatherRetry::Never,
            "stage{stage}: {error}"
        );
        assert_eq!(provider.stage.load(Ordering::SeqCst), stage + 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        provider.assert_retired();
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[test]
fn each_welch_continuation_failure_retains_cause_without_replay() {
    for stage in 0..=4 {
        let provider = Provider::new(Mode::Good, Upload::Good);
        provider.fail_stage.store(stage, Ordering::SeqCst);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let input = input(provider, NumericElementType::F64, false);
        let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "pwelch",
            &args(&input.handle),
            2,
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        let mut cause: &(dyn std::error::Error + 'static) = &error;
        while !cause.is::<std::io::Error>() {
            cause = cause.source().expect("original continuation cause");
        }
        assert_eq!(
            cause.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::BrokenPipe
        );
        assert_eq!(provider.stage.load(Ordering::SeqCst), stage + 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
        provider.assert_retired();
        assert_eq!(input.probe.retirements(), 0);
    }
}

#[test]
fn welch_native_psd_and_frequencies_preserve_class_and_output_intent() {
    for element in [NumericElementType::F32, NumericElementType::F64] {
        for explicit in [false, true] {
            for mode in [Mode::Good, Mode::Unsupported] {
                let provider = Provider::new(mode, Upload::Good);
                let _scope = ThreadProviderGuard::set(Some(provider));
                let input = input(provider, element, explicit);
                let output = block_on(runmat_runtime::call_builtin_async_with_outputs(
                    "pwelch",
                    &args(&input.handle),
                    2,
                ))
                .unwrap();
                let Value::OutputList(values) = output.value() else {
                    panic!("Welch two-output carrier")
                };
                assert_eq!(values.len(), 2);
                let (Value::GpuTensor(psd), Value::GpuTensor(frequency)) = (&values[0], &values[1])
                else {
                    panic!("resident Welch outputs")
                };
                for output in [psd, frequency] {
                    assert_eq!(output.shape(), &[3, 1]);
                    assert_eq!(output.physical().element_type(), element);
                    assert!(same_provider_instance(
                        provider_for_handle(output).unwrap(),
                        provider
                    ));
                    assert_eq!(handle_is_explicit(output), explicit);
                    assert!(!output.shares_allocation_with(&input.handle));
                }
                assert!(!psd.shares_allocation_with(frequency));
                let records = provider.produced.lock().unwrap();
                for (output, expected) in [
                    (psd, [0.0, 0.0, 0.0]),
                    (
                        frequency,
                        [0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
                    ),
                ] {
                    records
                        .iter()
                        .find_map(|probe| {
                            probe
                                .with_data(output, |data| {
                                    let values: Vec<f64> = match data {
                                        HostNumericDataView::F32(values) => {
                                            values.iter().map(|v| f64::from(*v)).collect()
                                        }
                                        HostNumericDataView::F64(values) => values.to_vec(),
                                        _ => panic!("floating Welch payload"),
                                    };
                                    for (actual, expected) in values.iter().zip(expected) {
                                        assert!((actual - expected).abs() < 2e-6);
                                    }
                                })
                                .ok()
                        })
                        .expect("owned Welch output payload");
                }
                drop(records);
                assert_eq!(
                    provider.downloads.load(Ordering::SeqCst),
                    usize::from(matches!(mode, Mode::Unsupported))
                );
                drop(output);
                provider.assert_retired();
                assert_eq!(input.probe.retirements(), 0);
            }
        }
    }
}

#[test]
fn welch_cancellation_retires_estimate_and_mean_without_touching_caller() {
    for stage in [0, 2] {
        let provider = Provider::new(Mode::Good, Upload::Good);
        provider.pending_stage.store(stage, Ordering::SeqCst);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let input = input(provider, NumericElementType::F64, true);
        let args = args(&input.handle);
        let mut future = Box::pin(runmat_runtime::call_builtin_async_with_outputs(
            "pwelch", &args, 2,
        ));
        let waker = noop_waker();
        assert!(future
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        assert_eq!(provider.stage.load(Ordering::SeqCst), stage + 1);
        drop(future);
        provider.assert_retired();
        input.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(input.probe.retirements(), 0);
    }
}
