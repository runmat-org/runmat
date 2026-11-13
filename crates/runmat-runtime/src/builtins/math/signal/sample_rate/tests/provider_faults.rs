use super::*;

#[test]
fn resident_scalar_controls_preflight_shape_class_and_retain_download_causes() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
    for (shape, storage) in [
        (vec![1, 2], GpuTensorStorage::Real),
        (vec![1, 1], GpuTensorStorage::ComplexInterleaved),
    ] {
        let control = provider
            .owner
            .allocate_zeroed(
                shape,
                runmat_accelerate_api::NumericElementType::F64,
                storage,
            )
            .expect("control")
            .handle;
        let before = provider.downloads.load(Ordering::Relaxed);
        let error = block_on(parse_factor(
            &Value::GpuTensor(control.clone()),
            UPSAMPLE_NAME,
        ))
        .expect_err("not a real scalar control");
        assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);
        assert_eq!(provider.downloads.load(Ordering::Relaxed), before);
        assert!(control.allocation_lease().validate_live().is_ok());
        control.release().expect("caller control remains owned");
    }
    let control = provider
        .upload(&HostTensorView {
            data: &[2.0],
            shape: &[1, 1],
        })
        .expect("scalar");
    for mode in [1, 2] {
        provider.download_mode.store(mode, Ordering::Relaxed);
        let error = block_on(parse_factor(
            &Value::GpuTensor(control.clone()),
            UPSAMPLE_NAME,
        ))
        .expect_err("provider download failure");
        assert_eq!(error.identifier(), SAMPLE_ERROR_INVALID_FACTOR.identifier);
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        let mut cursor: &(dyn std::error::Error + 'static) = &error;
        let mut retained = false;
        while let Some(source) = cursor.source() {
            retained |= source.to_string().contains(if mode == 1 {
                "unavailable"
            } else {
                "interrupted"
            });
            cursor = source;
        }
        assert!(
            retained,
            "original provider cause must survive owning mapping"
        );
        assert!(control.allocation_lease().validate_live().is_ok());
    }
    control.release().expect("caller scalar remains owned");
}

#[test]
fn resample_coefficient_upload_rejects_malformed_borrowed_and_failed_retirement_results() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &[1, 3],
        })
        .expect("source");
    for mode in 1..=6_u8 {
        provider.upload_mode.store(mode, Ordering::Relaxed);
        let filter_len = if mode == 5 { 11 } else { 3 };
        let options = ResampleOptions {
            filter: ResampleFilter::Host(vec![1.0; filter_len]),
            dim: None,
        };
        let before = provider.numeric_uploads.load(Ordering::Relaxed);
        let Err(error) = block_on(resample_gpu(&input, 2, 1, &options)) else {
            panic!("malformed or borrowed upload is terminal after scatter");
        };
        assert_eq!(provider.numeric_uploads.load(Ordering::Relaxed), before + 1);
        assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
        assert!(std::error::Error::source(&error).is_some());
        assert!(input.allocation_lease().validate_live().is_ok());
        let source = std::error::Error::source(&error).expect("typed upload cause");
        assert!(source
            .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
            .is_some_and(|execution| execution.is_after_effect()));
        if mode == 4 || mode == 5 {
            assert!(source.to_string().contains("aliasing a borrowed input"));
        }
        if mode == 6 {
            assert!(source.to_string().contains("retirement failure"));
        }
        provider.staged_scatter.lock().unwrap().take();
    }
    input
        .release()
        .expect("borrowed source survives every rejection");
}

#[test]
fn resample_coefficient_upload_failure_after_scatter_is_terminal_with_cause() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0],
            shape: &[1, 3],
        })
        .expect("source");
    let options = ResampleOptions {
        filter: ResampleFilter::Host(vec![0.0, 1.0, 0.0]),
        dim: None,
    };
    for mode in [7, 8] {
        provider.upload_mode.store(mode, Ordering::Relaxed);
        let downloads_before = provider.downloads.load(Ordering::Relaxed);
        let Err(error) = block_on(resample_gpu(&input, 2, 1, &options)) else {
            panic!("no replay after scatter");
        };
        assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
        assert!(std::error::Error::source(&error)
            .and_then(
                |source| source.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
            )
            .is_some_and(|execution| execution.is_after_effect()));
        assert_eq!(provider.downloads.load(Ordering::Relaxed), downloads_before);
        assert!(input.allocation_lease().validate_live().is_ok());
        provider.staged_scatter.lock().unwrap().take();
    }
    input.release().expect("borrowed source remains live");
}

#[test]
fn upsample_identity_rejects_unowned_input_before_alias_publication() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }
    let foreign = provider
        .foreign_owner
        .allocate_zeroed(
            vec![1, 4],
            runmat_accelerate_api::NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap()
        .handle;
    let retained = foreign.clone();
    let error = upsample_gpu(&foreign, 1, 0, UPSAMPLE_NAME)
        .expect_err("identity must authenticate its input owner");
    assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
    assert!(retained.allocation_lease().validate_live().is_ok());
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    let _ = retained.release();
    let _ = foreign.release();
}

#[test]
fn downsample_unknown_input_device_rejects_before_provider_work() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }

    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[1, 4],
        })
        .expect("input upload");
    let hostile = provider
        .foreign_owner
        .allocate_zeroed(
            input.shape().to_vec(),
            runmat_accelerate_api::NumericElementType::F64,
            GpuTensorStorage::Real,
        )
        .unwrap()
        .handle;
    let frees_before = provider.frees();

    let error = block_on(downsample_builtin(
        Value::GpuTensor(hostile),
        Value::Num(2.0),
        Vec::new(),
    ))
    .expect_err("unknown input device must reject before provider work");
    assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
    assert!(error.message().contains("no proven owner"));
    assert_eq!(provider.gathers.load(Ordering::Relaxed), 0);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), 0);
    assert_eq!(provider.frees(), frees_before);
    assert!(provider.is_live(&input));

    input.release().expect("release live input");
    assert_eq!(provider.live_count(), 0);
}

#[test]
fn downsample_rejects_hostile_provider_results_and_unphased_hook_failure() {
    let _guard = test_support::accel_test_lock();
    let provider = Box::leak(Box::new(HostileDownsampleProvider::new()));
    unsafe {
        runmat_accelerate_api::register_provider(provider);
    }

    for mode in 0..6_u8 {
        provider.mode.store(mode, Ordering::Relaxed);
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[1, 4],
            })
            .expect("input upload");
        let frees_before = provider.frees();
        let error = downsample_gpu(&input, 2, 0, DOWNSAMPLE_NAME)
            .expect_err("malformed provider result must be an explicit error");
        assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);

        assert_eq!(
            provider.frees(),
            frees_before + 1,
            "malformed owned output retires through its actual owner, including a foreign owner"
        );
        input.release().expect("free input");
        assert_eq!(provider.live_count(), 0);
    }

    provider.mode.store(7, Ordering::Relaxed);
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[1, 4],
        })
        .expect("alias input upload");
    let frees_before = provider.frees();
    let error = downsample_gpu(&input, 2, 0, DOWNSAMPLE_NAME)
        .expect_err("provider output alias must be an explicit error");
    assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
    assert_eq!(
        provider.frees(),
        frees_before,
        "aliased input must remain caller-owned"
    );
    assert!(provider.is_live(&input));
    input.release().expect("free aliased input");
    assert_eq!(provider.live_count(), 0);

    provider.mode.store(8, Ordering::Relaxed);
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[1, 4],
        })
        .expect("input upload");
    let frees_before = provider.frees();
    let error = downsample_gpu(&input, 2, 0, DOWNSAMPLE_NAME)
        .expect_err("a distinct view of borrowed input is not a fresh allocation");
    assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
    assert_eq!(provider.frees(), frees_before);
    assert!(provider.is_live(&input));
    input.release().expect("free view-aliased input");
    assert_eq!(provider.live_count(), 0);

    provider.mode.store(6, Ordering::Relaxed);
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0, 3.0, 4.0],
            shape: &[1, 4],
        })
        .expect("input upload");
    let downloads_before = provider.downloads.load(Ordering::Relaxed);
    let error = block_on(downsample_builtin(
        Value::GpuTensor(input.clone()),
        Value::Num(2.0),
        Vec::new(),
    ))
    .expect_err("unphased Unsupported must not authorize replay");
    assert_eq!(error.identifier(), SAMPLE_ERROR_GATHER_FAILED.identifier);
    assert_eq!(provider.downloads.load(Ordering::Relaxed), downloads_before);
    assert!(std::error::Error::source(&error).is_some());
    assert!(provider.is_live(&input));
    input.release().expect("free fallback input");
    assert_eq!(provider.live_count(), 0);
}
