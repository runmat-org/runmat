use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelNumericDownloadFuture,
    AccelProvider, GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct FailingDownloadProvider {
    owner: TestAllocationOwner,
    downloads: AtomicUsize,
}

impl AccelProvider for FailingDownloadProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "mnrfit gather fixture".into()
    }

    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("mnrfit fixture download failure") })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { anyhow::bail!("mnrfit fixture native download failure") })
    }
}

#[tokio::test]
async fn mnrfit_provider_failure_retains_cause_and_borrowed_lease() {
    let provider = Box::leak(Box::new(FailingDownloadProvider {
        owner: TestAllocationOwner::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = mnrfit_builtin(
        Value::GpuTensor(resident.handle.clone()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Vec::new(),
    )
    .await
    .expect_err("registered provider must fail the X download");
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 1);
    assert_eq!(error.context.builtin.as_deref(), Some("mnrfit"));
    assert!(error.message().contains("mnrfit fixture"));
    assert!(error.source.is_some());
    assert_eq!(resident.probe.retirements(), 0);
    drop(resident.handle);
    assert_eq!(resident.probe.retirements(), 1);
}
