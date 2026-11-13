//! Invalid calls must not touch an available, registered resident provider.
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostNumericTensorView, HostTensorView,
    NumericElementType, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct RejectingTransfers {
    owner: TestAllocationOwner,
    uploads: AtomicUsize,
    downloads: AtomicUsize,
}

impl AccelProvider for RejectingTransfers {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "distribution preflight transfer witness".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("invalid call attempted provider upload")
    }

    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        Err(anyhow::anyhow!("invalid call attempted numeric provider upload").into())
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("invalid call attempted provider download") })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("invalid call attempted numeric provider download") })
    }
}

#[test]
fn registered_distribution_provider_receives_no_transfers_for_invalid_calls() {
    let _guard = super::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(RejectingTransfers::default()));
    // This fixture is process-lived, and all global mutations are serialized.
    unsafe { runmat_accelerate_api::register_provider(provider) };
    let _thread = ThreadProviderGuard::set(Some(provider));
    let resident = provider
        .owner
        .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let selected = runmat_accelerate_api::provider_for_handle(&resident.handle).unwrap();
    assert!(runmat_accelerate_api::same_provider_instance(
        selected, provider
    ));
    super::shared::assert_invalid_calls(runmat_value::Value::GpuTensor(resident.handle.clone()));
    assert_eq!(provider.uploads.load(Ordering::SeqCst), 0);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(resident.probe.retirements(), 0);
}
