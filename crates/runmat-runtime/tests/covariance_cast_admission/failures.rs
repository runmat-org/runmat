use super::*;
use futures::FutureExt;
use runmat_runtime::RuntimeError;

fn has_cause<T: std::error::Error + 'static>(error: &RuntimeError) -> bool {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if source.is::<T>() {
            return true;
        }
        let Some(next) = source.source() else {
            return false;
        };
        source = next;
    }
}

fn io_cause_kind(error: &RuntimeError) -> Option<std::io::ErrorKind> {
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(io) = source.downcast_ref::<std::io::Error>() {
            return Some(io.kind());
        }
        source = source.source()?;
    }
}

fn input(provider: &Provider) -> runmat_accelerate_api::test_support::TestAllocation {
    provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn certified_widening_refusal_adapts_only_the_produced_f32_covariance() {
    let provider = Provider::new(Mode::WidenUnsupported);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original = provider
        .owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    *provider.input_probe.lock().unwrap() = Some(original.probe.clone());
    let result = block_on(call_builtin_async_with_outputs(
        "cov",
        &[Value::GpuTensor(original.handle.clone())],
        1,
    ))
    .unwrap();
    let output = match result.value() {
        Value::OutputList(values) if values.len() == 1 => &values[0],
        value => value,
    };
    let Value::GpuTensor(output) = output else {
        panic!("covariance should restore the selected F64 result class");
    };
    assert_eq!(handle_precision(output), Some(ProviderPrecision::F64));
    assert_eq!(output.shape(), &[2, 2]);
    assert!(!output.shares_allocation_with(&original.handle));
    assert_eq!(provider.covariance_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.casts.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.input_downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 1);
    let downloaded = block_on(provider.download_numeric(output)).unwrap();
    assert_eq!(downloaded.shape, vec![2, 2]);
    assert_eq!(downloaded.data, HostNumericDataOwned::F64(vec![0.5; 4]));
    drop(result);
    let outputs = provider.outputs.lock().unwrap();
    assert!(!outputs.is_empty());
    assert!(outputs.iter().all(|probe| probe.retirements() == 1));
    original.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(original.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn produced_result_download_and_restoration_failures_never_recompute_covariance() {
    for (mode, explicit, expected_uploads) in [
        (Mode::ProducedDownloadFault, false, 0),
        (Mode::RestoreUnsupported, true, 1),
        (Mode::RestoreFault, false, 1),
    ] {
        let provider = Provider::new(mode);
        let _selected = ThreadProviderGuard::set(Some(provider));
        let original = input(provider);
        *provider.input_probe.lock().unwrap() = Some(original.probe.clone());
        let selected = if explicit {
            original
                .handle
                .with_provenance(GpuHandleProvenance::Explicit)
                .unwrap()
        } else {
            original.handle.clone()
        };
        let error = block_on(call_builtin_async_with_outputs(
            "cov",
            &[Value::GpuTensor(selected)],
            1,
        ))
        .expect_err("a failed adaptation must not restart covariance");

        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never, "{mode:?}");
        assert_eq!(provider.covariance_calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.corrcoef_calls.load(Ordering::SeqCst), 0);
        assert_eq!(provider.casts.load(Ordering::SeqCst), 1);
        assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
        assert_eq!(provider.input_downloads.load(Ordering::SeqCst), 0);
        assert_eq!(provider.uploads.load(Ordering::SeqCst), expected_uploads);
        match mode {
            Mode::RestoreUnsupported => {
                assert!(has_cause::<UnsupportedProviderOperation>(&error));
            }
            Mode::ProducedDownloadFault | Mode::RestoreFault => {
                assert!(has_cause::<std::io::Error>(&error));
                assert_eq!(io_cause_kind(&error), Some(std::io::ErrorKind::BrokenPipe));
            }
            _ => unreachable!(),
        }
        let outputs = provider.outputs.lock().unwrap();
        assert!(!outputs.is_empty());
        assert!(outputs.iter().all(|probe| probe.retirements() == 1));
        original.handle.allocation_lease().validate_live().unwrap();
        assert_eq!(original.probe.retirements(), 0);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn abandoning_pending_produced_result_download_retires_only_owned_temporaries() {
    let provider = Provider::new(Mode::PendingProducedDownload);
    let _selected = ThreadProviderGuard::set(Some(provider));
    let original = input(provider);
    *provider.input_probe.lock().unwrap() = Some(original.probe.clone());
    let arguments = [Value::GpuTensor(original.handle.clone())];
    let pending = call_builtin_async_with_outputs("cov", &arguments, 1);
    assert!(pending.now_or_never().is_none());
    assert_eq!(provider.covariance_calls.load(Ordering::SeqCst), 1);
    assert_eq!(provider.casts.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(provider.input_downloads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    let outputs = provider.outputs.lock().unwrap();
    assert!(!outputs.is_empty());
    assert!(outputs.iter().all(|probe| probe.retirements() == 1));
    original.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(original.probe.retirements(), 0);
}
