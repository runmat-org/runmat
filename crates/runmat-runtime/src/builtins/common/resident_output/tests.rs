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
    Mutex, OnceLock,
};

struct AliasProvider {
    inner: InProcessProvider,
    alias: Mutex<Option<GpuTensorHandle>>,
    uploads: AtomicUsize,
    frees: AtomicUsize,
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
    fn upload_numeric(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        if let Some(handle) = self.alias.lock().unwrap().as_ref() {
            return Ok(handle.clone());
        }
        self.inner.upload_numeric(host)
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
    fn free(&self, handle: &GpuTensorHandle) -> anyhow::Result<()> {
        self.frees.fetch_add(1, Ordering::Relaxed);
        self.inner.free(handle)
    }
}

fn provider() -> &'static AliasProvider {
    static PROVIDER: OnceLock<AliasProvider> = OnceLock::new();
    let provider = PROVIDER.get_or_init(|| AliasProvider {
        inner: InProcessProvider::new(),
        alias: Mutex::new(None),
        uploads: AtomicUsize::new(0),
        frees: AtomicUsize::new(0),
    });
    *provider.alias.lock().unwrap() = None;
    provider.uploads.store(0, Ordering::Relaxed);
    provider.frees.store(0, Ordering::Relaxed);
    provider
}

#[test]
fn unknown_source_owner_never_uses_current_provider() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider();
    let _provider = ThreadProviderGuard::set(Some(provider));
    let source = GpuTensorHandle::new(vec![1, 1], u32::MAX, 777);
    assert!(restore_to_source_provider(Value::Num(1.0), Some(&source)).is_err());
    assert_eq!(provider.uploads.load(Ordering::Relaxed), 0);
    assert_eq!(provider.frees.load(Ordering::Relaxed), 0);
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
        assert_eq!(
            provider.frees.load(Ordering::Relaxed),
            0,
            "{name}: borrowed aliases remain alive"
        );
        assert!(block_on(provider.inner.download_integer(&left)).is_ok());
        assert!(block_on(provider.inner.download_integer(&right)).is_ok());
        provider.inner.free(&left).unwrap();
        provider.inner.free(&right).unwrap();
    }
    assert_eq!(provider.uploads.load(Ordering::Relaxed), 6);
}

#[test]
fn default_wrapper_protects_source_and_preserves_explicit_provenance() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    let provider = provider();
    let _provider = ThreadProviderGuard::set(Some(provider));
    let mut source = provider
        .inner
        .upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .unwrap();
    runmat_accelerate_api::mark_handle_explicit(&mut source);
    *provider.alias.lock().unwrap() = Some(source.clone());
    assert!(restore_to_source_provider(Value::Num(2.0), Some(&source)).is_err());
    assert_eq!(provider.frees.load(Ordering::Relaxed), 0);
    *provider.alias.lock().unwrap() = None;
    let Value::GpuTensor(output) =
        restore_to_source_provider(Value::Num(2.0), Some(&source)).unwrap()
    else {
        panic!("resident result")
    };
    assert_eq!(
        runmat_accelerate_api::handle_provenance(&output),
        runmat_accelerate_api::handle_provenance(&source)
    );
    provider.inner.free(&source).unwrap();
    provider.inner.free(&output).unwrap();
}
