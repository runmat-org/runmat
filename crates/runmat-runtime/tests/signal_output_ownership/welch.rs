use super::*;

fn args(input: &GpuTensorHandle) -> Vec<Value> {
    vec![
        Value::GpuTensor(input.clone()),
        Value::Num(2.0),
        Value::Num(0.0),
        Value::Num(4.0),
    ]
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
                let Value::OutputList(values) = output else {
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
                drop(values);
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
