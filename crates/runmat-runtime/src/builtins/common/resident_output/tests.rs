use super::*;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, HostIntegerDataView,
    HostIntegerTensorView, HostNumericTensorView, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

struct AliasProvider {
    inner: InProcessProvider,
    alias: Mutex<Option<GpuTensorHandle>>,
    uploads: AtomicUsize,
}
impl AccelProvider for AliasProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        if let Some(handle) = self.alias.lock().unwrap().as_ref() {
            return Ok(handle.clone());
        }
        self.inner.upload_numeric_raw(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.inner.download_numeric(handle)
    }
}

fn provider() -> &'static AliasProvider {
    Box::leak(Box::new(AliasProvider {
        inner: InProcessProvider::new(),
        alias: Mutex::new(None),
        uploads: AtomicUsize::new(0),
    }))
}

#[test]
fn unknown_source_owner_never_uses_current_provider() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider();
    let _provider = ThreadProviderGuard::set(Some(provider));
    let unregistered_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let allocation = unregistered_owner
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let source = &allocation.handle;
    assert!(
        restore_to_source_provider_protecting(Value::Num(1.0), Some(source), &[source]).is_err()
    );
    assert_eq!(provider.uploads.load(Ordering::Relaxed), 0);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn bitwise_restoration_protects_other_resident_operands_and_controls() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider();
    let _provider = ThreadProviderGuard::set(Some(provider));
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for name in ["bitand", "bitor", "bitxor", "bitget", "bitset", "bitshift"] {
        let left = provider
            .inner
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&[3, 7]),
                shape: &[1, 2],
            })
            .unwrap();
        let right = provider
            .inner
            .upload_integer(&HostIntegerTensorView {
                data: HostIntegerDataView::U8(&[1, 2]),
                shape: &[1, 2],
            })
            .unwrap();
        *provider.alias.lock().unwrap() = Some(right.clone());
        let result = block_on(crate::dispatcher::call_builtin_async(
            name,
            &[
                Value::GpuTensor(left.clone()),
                Value::GpuTensor(right.clone()),
            ],
        ));
        assert!(result.is_err(), "{name}: upload alias must be rejected");
        assert!(block_on(provider.inner.download_integer(&left)).is_ok());
        assert!(block_on(provider.inner.download_integer(&right)).is_ok());
        let left_weak = left.downgrade();
        let right_weak = right.downgrade();
        *provider.alias.lock().unwrap() = None;
        drop((left, right));
        assert!(left_weak.acquire().is_err());
        assert!(right_weak.acquire().is_err());
    }
    assert_eq!(provider.uploads.load(Ordering::Relaxed), 6);
}

#[test]
fn default_wrapper_protects_source_and_preserves_explicit_provenance() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider();
    let _provider = ThreadProviderGuard::set(Some(provider));
    let source = provider
        .inner
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap()
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    *provider.alias.lock().unwrap() = Some(source.clone());
    assert!(
        restore_to_source_provider_protecting(Value::Num(2.0), Some(&source), &[&source]).is_err()
    );
    source.allocation_lease().validate_live().unwrap();
    *provider.alias.lock().unwrap() = None;
    let Value::GpuTensor(output) =
        restore_to_source_provider_protecting(Value::Num(2.0), Some(&source), &[&source]).unwrap()
    else {
        panic!("resident result")
    };
    assert_eq!(
        runmat_accelerate_api::handle_provenance(&output),
        runmat_accelerate_api::handle_provenance(&source)
    );
    let source_weak = source.downgrade();
    let output_weak = output.downgrade();
    drop((source, output));
    assert!(source_weak.acquire().is_err());
    assert!(output_weak.acquire().is_err());
}
