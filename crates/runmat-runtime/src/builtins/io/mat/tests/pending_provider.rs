use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostTensorView, ProviderPrecision,
};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(in crate::builtins::io::mat) struct PendingDownloadProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    requests: AtomicUsize,
    abandoned: AtomicUsize,
}

impl PendingDownloadProvider {
    pub(in crate::builtins::io::mat) fn new() -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            requests: AtomicUsize::new(0),
            abandoned: AtomicUsize::new(0),
        }
    }

    pub(in crate::builtins::io::mat) fn resident(&self) -> GpuTensorHandle {
        self.upload(&HostTensorView {
            data: &[42.0],
            shape: &[1, 1],
        })
        .expect("authentic pending-provider allocation")
    }

    pub(in crate::builtins::io::mat) fn requests(&self) -> usize {
        self.requests.load(Ordering::SeqCst)
    }

    pub(in crate::builtins::io::mat) fn abandoned(&self) -> usize {
        self.abandoned.load(Ordering::SeqCst)
    }
}

struct PendingDownload<'a>(&'a AtomicUsize);
impl Drop for PendingDownload<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl AccelProvider for PendingDownloadProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let _pending = PendingDownload(&self.abandoned);
            std::future::pending().await
        })
    }

    fn download_numeric<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelNumericDownloadFuture<'a> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let _pending = PendingDownload(&self.abandoned);
            std::future::pending().await
        })
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
