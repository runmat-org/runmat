//! Network-domain provider witness for pre-effect admission tests.

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostTensorView, ProviderPrecision,
};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct CountingDownloadProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

impl CountingDownloadProvider {
    pub(crate) fn new() -> Self {
        Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
            downloads: AtomicUsize::new(0),
        }
    }

    pub(crate) fn resident(&self) -> GpuTensorHandle {
        self.upload(&HostTensorView {
            data: &[1.0],
            shape: &[1, 1],
        })
        .expect("authentic network provider allocation")
    }

    pub(crate) fn downloads(&self) -> usize {
        self.downloads.load(Ordering::SeqCst)
    }
}

impl AccelProvider for CountingDownloadProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(handle)
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
