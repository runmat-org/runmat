use super::*;

#[test]
fn protected_provider_siblings_cannot_change_automatic_caller_intent() {
    for count in [1, 2] {
        let provider = Provider::new(Mode::ExplicitSibling, Upload::Good);
        let _scope = ThreadProviderGuard::set(Some(provider));
        let input = input(provider, NumericElementType::F64, false);
        let output = block_on(runmat_runtime::call_builtin_async_with_outputs(
            "envelope",
            &[Value::GpuTensor(input.handle.clone())],
            count,
        ))
        .unwrap();
        let Value::OutputList(values) = output else {
            panic!("envelope output list")
        };
        assert_eq!(values.len(), count);
        for value in &values {
            let Value::GpuTensor(handle) = value else {
                panic!("resident output")
            };
            assert_eq!(
                handle_provenance(handle),
                Some(GpuHandleProvenance::Automatic)
            );
            assert!(!handle.shares_allocation_with(&input.handle));
        }
        drop(values);
        provider.assert_retired();
        assert_eq!(input.probe.retirements(), 0);
        assert!(!handle_is_explicit(&input.handle));
    }
}

#[test]
fn malformed_lower_retires_already_admitted_upper() {
    let provider = Provider::new(Mode::LowerShape, Upload::Good);
    let _scope = ThreadProviderGuard::set(Some(provider));
    let input = input(provider, NumericElementType::F64, true);
    let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "envelope",
        &[Value::GpuTensor(input.handle.clone())],
        2,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    assert_eq!(provider.produced.lock().unwrap().len(), 2);
    provider.assert_retired();
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn second_fallback_upload_failure_retires_first_owned_output() {
    let provider = Provider::new(Mode::Unsupported, Upload::SecondFailure);
    let _scope = ThreadProviderGuard::set(Some(provider));
    let input = input(provider, NumericElementType::F32, true);
    let error = block_on(runmat_runtime::call_builtin_async_with_outputs(
        "envelope",
        &[Value::GpuTensor(input.handle.clone())],
        2,
    ))
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<std::io::Error>() {
        cause = cause.source().expect("second upload cause");
    }
    assert_eq!(
        cause.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::BrokenPipe
    );
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 2);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.hooks.load(Ordering::SeqCst), 1);
    assert_eq!(provider.produced.lock().unwrap().len(), 1);
    provider.assert_retired();
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
}
