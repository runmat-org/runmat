use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, GpuTensorStorage, HostNumericTensorView, HostTensorView,
    NumericElementType, ThreadProviderGuard,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
struct Provider {
    owner: TestAllocationOwner,
    uploaded: Mutex<Vec<RetirementProbe>>,
    calls: AtomicUsize,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "reduction pair owning restoration".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("untyped upload forbidden")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected download") })
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 1 {
            return Err(anyhow::anyhow!("second output failed").into());
        }
        let output = self.owner.allocate_zeroed(
            host.shape.to_vec(),
            host.data.element_type(),
            host.storage,
        )?;
        self.uploaded.lock().unwrap().push(output.probe);
        Ok(output.handle)
    }
}

#[test]
fn second_restore_failure_consumes_first_result_without_releasing_source() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        uploaded: Mutex::new(Vec::new()),
        calls: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = restore_pair(
        &source.handle,
        Value::Num(1.0),
        Value::Num(2.0),
        2,
        "bounds",
        &[&source.handle],
    )
    .unwrap_err();
    assert!(error.message().contains("second output failed"));
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    let uploaded = provider.uploaded.lock().unwrap();
    assert_eq!(uploaded.len(), 1);
    assert_eq!(uploaded[0].retirements(), 1);
    assert_eq!(source.probe.retirements(), 0);
    source.handle.allocation_lease().validate_live().unwrap();
}
