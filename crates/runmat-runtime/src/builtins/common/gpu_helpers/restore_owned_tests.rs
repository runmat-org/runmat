use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, HostLogicalTensorView, HostTensorView,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
enum NumericUpload {
    Success,
    Unsupported,
    UnprovenUnsupported,
    AfterEffectUnsupported,
    Execution,
}
#[derive(Clone, Copy)]
enum LogicalUpload {
    Success,
    Unsupported,
    UnprovenUnsupported,
    AfterEffectUnsupported,
}
struct Provider {
    owner: TestAllocationOwner,
    numeric_upload: NumericUpload,
    logical_uploads: AtomicUsize,
    unavailable_upload: AtomicBool,
    output_probe: Mutex<Option<runmat_accelerate_api::test_support::RetirementProbe>>,
    logical_upload: Mutex<LogicalUpload>,
}
impl Provider {
    fn new(numeric_upload: NumericUpload) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            numeric_upload,
            logical_uploads: AtomicUsize::new(0),
            unavailable_upload: AtomicBool::new(false),
            output_probe: Mutex::new(None),
            logical_upload: Mutex::new(LogicalUpload::Success),
        }))
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "class restoration owner".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("must use typed upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("restoration must not download") })
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.unavailable_upload.load(Ordering::SeqCst)
            || matches!(self.numeric_upload, NumericUpload::Success)
        {
            let allocation = self.owner.allocate_zeroed(
                host.shape.to_vec(),
                host.data.element_type(),
                host.storage,
            )?;
            if self.unavailable_upload.load(Ordering::SeqCst) {
                allocation.probe.mark_unavailable();
            }
            *self.output_probe.lock().unwrap() = Some(allocation.probe);
            return Ok(allocation.handle);
        }
        let error = match self.numeric_upload {
            NumericUpload::Success => unreachable!("successful upload returned above"),
            NumericUpload::Unsupported
            | NumericUpload::UnprovenUnsupported
            | NumericUpload::AfterEffectUnsupported => {
                runmat_accelerate_api::unsupported_provider_operation("fixture unsupported")
            }
            NumericUpload::Execution => anyhow::anyhow!("fixture execution failed"),
        };
        let error = runmat_accelerate_api::ProviderExecutionError::from(error);
        Err(match self.numeric_upload {
            NumericUpload::UnprovenUnsupported => error.with_unproven_effects(),
            NumericUpload::AfterEffectUnsupported => error.after_effect(),
            _ => error,
        })
    }
    fn upload_logical_raw(
        &self,
        view: &HostLogicalTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        match *self.logical_upload.lock().unwrap() {
            LogicalUpload::Success => {
                self.logical_uploads.fetch_add(1, Ordering::SeqCst);
                Ok(self
                    .owner
                    .allocate_logical(view.data.to_vec(), view.shape.to_vec())
                    .map_err(|error| {
                        runmat_accelerate_api::ProviderExecutionError::from(error)
                            .with_unproven_effects()
                    })?
                    .handle)
            }
            behavior => {
                self.logical_uploads.fetch_add(1, Ordering::SeqCst);
                let error = runmat_accelerate_api::ProviderExecutionError::from(
                    runmat_accelerate_api::unsupported_provider_operation(
                        "fixture logical representation unavailable",
                    ),
                );
                Err(match behavior {
                    LogicalUpload::Unsupported => error,
                    LogicalUpload::UnprovenUnsupported => error.with_unproven_effects(),
                    LogicalUpload::AfterEffectUnsupported => error.after_effect(),
                    LogicalUpload::Success => unreachable!(),
                })
            }
        }
    }
}

#[test]
fn host_computed_real_scalar_restore_is_exact_and_never_replays() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    for (precision, element) in [
        (ProviderPrecision::F32, NumericElementType::F32),
        (ProviderPrecision::F64, NumericElementType::F64),
    ] {
        let provider = Provider::new(NumericUpload::Success);
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let source = provider
            .owner
            .allocate_zeroed(vec![2, 2], element, GpuTensorStorage::Real)
            .unwrap();
        let Value::GpuTensor(output) = restore_real_scalar_after_host(
            &source.handle,
            provider,
            2.5,
            precision,
            "scalar-restore-test",
            |message| build_runtime_error(message).build(),
        )
        .unwrap() else {
            panic!("typed scalar restore must remain resident");
        };
        assert_eq!(output.shape(), [1, 1]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(precision)
        );
        assert!(!output.shares_allocation_with(&source.handle));
        source.handle.allocation_lease().validate_live().unwrap();
    }

    let provider = Provider::new(NumericUpload::Unsupported);
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let failure = restore_real_scalar_after_host(
        &source.handle,
        provider,
        2.5,
        ProviderPrecision::F64,
        "scalar-restore-test",
        |message| build_runtime_error(message).build(),
    )
    .unwrap_err();
    assert_eq!(failure.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let mut cause: &(dyn std::error::Error + 'static) = &failure;
    let mut found_unproven = false;
    while let Some(next) = cause.source() {
        if let Some(execution) =
            next.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
        {
            found_unproven = !execution.is_unsupported() && !execution.is_after_effect();
        }
        cause = next;
    }
    assert!(found_unproven, "prior host work must forbid replay");
    source.handle.allocation_lease().validate_live().unwrap();
}

#[test]
fn numeric_restoration_preserves_rank_owner_class_and_allocation_independence() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Provider::new(NumericUpload::Success);
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();

    for (value, expected_shape, expected_storage) in [
        (
            Value::Tensor(Tensor::new(vec![7.0], Vec::new()).unwrap()),
            Vec::new(),
            GpuTensorStorage::Real,
        ),
        (
            Value::ComplexTensor(
                ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2]).unwrap(),
            ),
            vec![2],
            GpuTensorStorage::ComplexInterleaved,
        ),
    ] {
        let Value::GpuTensor(output) =
            restore_class_preserving_value(&source.handle, value, "restore-test").unwrap()
        else {
            panic!("successful restoration must remain resident");
        };
        assert_eq!(output.shape(), expected_shape.as_slice());
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            expected_storage
        );
        assert_eq!(
            runmat_accelerate_api::handle_precision(&output),
            Some(ProviderPrecision::F64)
        );
        assert_eq!(output.device_id(), source.handle.device_id());
        assert!(!output.shares_allocation_with(&source.handle));
        let output_lease = output.allocation_lease().downgrade();
        drop(output);
        assert!(output_lease.acquire().is_none());
        assert_eq!(source.probe.retirements(), 0);
    }

    drop(source.handle);
    assert_eq!(source.probe.retirements(), 1);
}

#[test]
fn unavailable_uploaded_result_is_terminal_and_retains_acquisition_cause() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Provider::new(NumericUpload::Execution);
    provider.unavailable_upload.store(true, Ordering::SeqCst);
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = restore_class_preserving_value(
        &source.handle,
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        "restore-test",
    )
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    while !cause.is::<runmat_accelerate_api::allocation::AllocationAcquisitionError>() {
        cause = cause.source().expect("retained acquisition cause");
    }
    assert_eq!(
        cause.downcast_ref::<runmat_accelerate_api::allocation::AllocationAcquisitionError>(),
        Some(&runmat_accelerate_api::allocation::AllocationAcquisitionError::NotLive)
    );
    assert_eq!(
        provider
            .output_probe
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .retirements(),
        1
    );
    source.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(source.probe.retirements(), 0);
}

#[test]
fn floating_restoration_requires_proven_automatic_intent_and_pre_effect_unsupported() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    use runmat_accelerate_api::GpuHandleProvenance::{Automatic, Explicit};
    for value in [
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![1, 2]).unwrap()),
    ] {
        for provenance in [None, Some(Automatic), Some(Explicit)] {
            for behavior in [
                NumericUpload::Unsupported,
                NumericUpload::UnprovenUnsupported,
                NumericUpload::AfterEffectUnsupported,
                NumericUpload::Execution,
            ] {
                let provider = Provider::new(behavior);
                let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
                let source = provider
                    .owner
                    .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                    .unwrap();
                let source_view = provenance.map_or_else(
                    || source.handle.clone(),
                    |intent| source.handle.with_provenance(intent).unwrap(),
                );
                let result =
                    restore_class_preserving_value(&source_view, value.clone(), "restore-test");
                if provenance == Some(Automatic) && matches!(behavior, NumericUpload::Unsupported) {
                    assert_eq!(result.unwrap(), value);
                } else {
                    let error = result.expect_err("unproven intent or effect is terminal");
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                    assert!(std::error::Error::source(&error).is_some());
                }
                assert_eq!(source.probe.retirements(), 0);
                drop(source_view);
                drop(source.handle);
                assert_eq!(source.probe.retirements(), 1);
            }
        }
    }
}

#[test]
fn logical_restoration_uses_owner_admitted_byte_upload_not_numeric_relabel() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Provider::new(NumericUpload::Execution);
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let result = restore_class_preserving_value(
        &source.handle,
        Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
        "restore-test",
    )
    .unwrap();
    let Value::GpuTensor(result) = result else {
        panic!("logical result must remain resident")
    };
    assert!(runmat_accelerate_api::handle_is_logical(&result));
    assert_eq!(result.physical().element_type(), NumericElementType::U8);
    assert_eq!(result.shape(), &[1, 2]);
    assert!(!result.shares_allocation_with(&source.handle));
    assert_eq!(provider.logical_uploads.load(Ordering::SeqCst), 1);
    drop(result);
    assert_eq!(source.probe.retirements(), 0);
    drop(source.handle);
    assert_eq!(source.probe.retirements(), 1);
}

#[test]
fn logical_restoration_falls_back_only_for_certified_automatic_unsupported() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    use runmat_accelerate_api::GpuHandleProvenance::{Automatic, Explicit};
    for provenance in [None, Some(Automatic), Some(Explicit)] {
        for behavior in [
            LogicalUpload::Unsupported,
            LogicalUpload::UnprovenUnsupported,
            LogicalUpload::AfterEffectUnsupported,
        ] {
            let provider = Provider::new(NumericUpload::Success);
            *provider.logical_upload.lock().unwrap() = behavior;
            let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
            let source = provider
                .owner
                .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let source_view = provenance.map_or_else(
                || source.handle.clone(),
                |intent| source.handle.with_provenance(intent).unwrap(),
            );
            let logical = Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap());
            let result =
                restore_class_preserving_value(&source_view, logical.clone(), "restore-test");
            if provenance == Some(Automatic) && matches!(behavior, LogicalUpload::Unsupported) {
                assert_eq!(result.unwrap(), logical);
            } else {
                let error =
                    result.expect_err("unproven, post-effect, and explicit failures are terminal");
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(std::error::Error::source(&error).is_some());
            }
            assert_eq!(provider.logical_uploads.load(Ordering::SeqCst), 1);
            assert!(source.handle.allocation_lease().validate_live().is_ok());
            drop(source_view);
            drop(source.handle);
            assert_eq!(source.probe.retirements(), 1);
        }
    }
}

#[test]
fn fallback_requires_every_output_intent_operand_to_be_proven_automatic() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    use runmat_accelerate_api::GpuHandleProvenance::{Automatic, Explicit};
    for value in [
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::LogicalArray(LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
    ] {
        for intent in [None, Some(Automatic), Some(Explicit)] {
            let provider = Provider::new(NumericUpload::Unsupported);
            *provider.logical_upload.lock().unwrap() = LogicalUpload::Unsupported;
            let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
            let source = provider
                .owner
                .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let operand = provider
                .owner
                .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let source_view = source.handle.with_provenance(Automatic).unwrap();
            let intent_view = intent.map_or_else(
                || operand.handle.clone(),
                |provenance| operand.handle.with_provenance(provenance).unwrap(),
            );
            let result = restore_class_preserving_value_with_operands(
                &source_view,
                value.clone(),
                "restore-test",
                RestoreOperandContext::new(&[], &[&intent_view]),
            );
            if intent == Some(Automatic) {
                assert_eq!(result.unwrap(), value);
            } else {
                let error = result.expect_err("unknown or explicit intent cannot fall back");
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            }
            assert!(source.handle.allocation_lease().validate_live().is_ok());
            assert!(operand.handle.allocation_lease().validate_live().is_ok());
            drop((source_view, intent_view));
            drop((source.handle, operand.handle));
            assert_eq!(source.probe.retirements(), 1);
            assert_eq!(operand.probe.retirements(), 1);
        }
    }
}
