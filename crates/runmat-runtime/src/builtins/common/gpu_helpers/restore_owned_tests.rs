use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, HostLogicalTensorView, HostTensorView,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
enum NumericFailure {
    Unsupported,
    Execution,
}
struct Provider {
    owner: TestAllocationOwner,
    failure: NumericFailure,
    logical_uploads: AtomicUsize,
    unavailable_upload: AtomicBool,
    output_probe: Mutex<Option<runmat_accelerate_api::test_support::RetirementProbe>>,
}
impl Provider {
    fn new(failure: NumericFailure) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            failure,
            logical_uploads: AtomicUsize::new(0),
            unavailable_upload: AtomicBool::new(false),
            output_probe: Mutex::new(None),
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
    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        if self.unavailable_upload.load(Ordering::SeqCst) {
            let allocation = self.owner.allocate_zeroed(
                host.shape.to_vec(),
                host.data.element_type(),
                host.storage,
            )?;
            allocation.probe.mark_unavailable();
            *self.output_probe.lock().unwrap() = Some(allocation.probe);
            return Ok(allocation.handle);
        }
        Err(match self.failure {
            NumericFailure::Unsupported => {
                runmat_accelerate_api::unsupported_provider_operation("fixture unsupported")
            }
            NumericFailure::Execution => anyhow::anyhow!("fixture execution failed"),
        })
    }
    fn upload_logical_raw(&self, view: &HostLogicalTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.logical_uploads.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .owner
            .allocate_logical(view.data.to_vec(), view.shape.to_vec())?
            .handle)
    }
}

#[test]
fn unavailable_uploaded_result_is_terminal_and_retains_acquisition_cause() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Provider::new(NumericFailure::Execution);
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
fn automatic_restoration_falls_back_only_for_typed_unsupported() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    for failure in [NumericFailure::Unsupported, NumericFailure::Execution] {
        let provider = Provider::new(failure);
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let source = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let value = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let result = restore_class_preserving_value(&source.handle, value.clone(), "restore-test");
        match provider.failure {
            NumericFailure::Unsupported => assert_eq!(result.unwrap(), value),
            NumericFailure::Execution => {
                let error = result.unwrap_err();
                assert!(error.message().contains("fixture execution failed"));
                assert!(std::error::Error::source(&error).is_some());
            }
        }
        assert_eq!(source.probe.retirements(), 0);
        drop(source.handle);
        assert_eq!(source.probe.retirements(), 1);
    }
}

#[test]
fn logical_restoration_uses_owner_admitted_byte_upload_not_numeric_relabel() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider = Provider::new(NumericFailure::Execution);
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
