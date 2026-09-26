//! A real in-process owner with controllable download suspension/failure.
use super::*;
use runmat_accelerate_api::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct DownloadOwner {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    mode: AtomicUsize,
    downloads: AtomicUsize,
    abandoned: AtomicUsize,
}
struct PendingRead<'a>(&'a AtomicUsize);
impl Drop for PendingRead<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl AccelProvider for DownloadOwner {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_numeric(host)
    }
    fn download<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(h)
    }
    fn download_numeric<'a>(&'a self, h: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            match self.mode.load(Ordering::SeqCst) {
                1 => {
                    let _read = PendingRead(&self.abandoned);
                    futures::future::pending::<()>().await;
                    unreachable!()
                }
                2 => anyhow::bail!("injected download failure"),
                _ => self.inner.download_numeric(h).await,
            }
        })
    }
    fn free(&self, h: &GpuTensorHandle) -> anyhow::Result<()> {
        self.inner.free(h)
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
}

#[test]
fn nested_download_cancellation_and_failure_preserve_borrowed_source() {
    use std::future::Future;
    let _lock = test_support::accel_test_lock();
    let owner = Box::leak(Box::new(DownloadOwner {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        mode: AtomicUsize::new(1),
        downloads: AtomicUsize::new(0),
        abandoned: AtomicUsize::new(0),
    }));
    let _provider = ThreadProviderGuard::set(Some(owner));
    let tensor = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
    let handle = crate::builtins::common::gpu_helpers::upload_tensor(owner, &tensor).unwrap();
    let value = Value::Cell(CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1).unwrap());
    let mut future = Box::pin(must_be_nonmissing::evaluate(vec![value.clone()]));
    let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
    assert!(future.as_mut().poll(&mut cx).is_pending());
    assert_eq!(owner.downloads.load(Ordering::SeqCst), 1);
    drop(future);
    assert_eq!(owner.abandoned.load(Ordering::SeqCst), 1);
    owner.mode.store(2, Ordering::SeqCst);
    let error =
        futures::executor::block_on(must_be_nonmissing::evaluate(vec![value.clone()])).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:gpu:DownloadFailed"));
    owner.mode.store(0, Ordering::SeqCst);
    assert!(futures::executor::block_on(owner.download_numeric(&handle)).is_ok());
    // Nested explicit intent gates before any pending/content-dependent access.
    let explicit = handle
        .clone()
        .with_provenance(GpuHandleProvenance::Explicit);
    let nested = Value::Cell(CellArray::new(vec![Value::GpuTensor(explicit)], 1, 1).unwrap());
    let reads = owner.downloads.load(Ordering::SeqCst);
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = futures::executor::block_on(must_be_non_nan::evaluate(vec![nested])).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:mustBeNonNanResidentInput")
    );
    assert_eq!(owner.downloads.load(Ordering::SeqCst), reads);
    owner.free(&handle).unwrap();
    clear_handle_metadata(&handle);
}
