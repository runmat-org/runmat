//! Provider that returns a borrowed control allocation from restoration.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Mutex,
};

use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostNumericDataView, HostNumericTensorView, HostTensorView,
};

pub(crate) struct RestoreProbeProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    alias: Mutex<Option<GpuTensorHandle>>,
    fail_upload: AtomicBool,
    downloads: AtomicUsize,
}

impl RestoreProbeProvider {
    pub(crate) fn new() -> &'static Self {
        Box::leak(Box::new(Self {
            inner: runmat_accelerate::simple_provider::InProcessProvider::default(),
            alias: Mutex::new(None),
            fail_upload: AtomicBool::new(false),
            downloads: AtomicUsize::new(0),
        }))
    }

    pub(crate) fn input(&self, values: &[f64]) -> GpuTensorHandle {
        self.inner
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F64(values),
                shape: &[1, values.len()],
                storage: GpuTensorStorage::Real,
            })
            .expect("fixture input upload")
    }

    pub(crate) fn alias_on_restore(&self, handle: &GpuTensorHandle) {
        *self.alias.lock().unwrap() = Some(handle.clone());
    }

    pub(crate) fn fail_on_restore(&self) {
        self.fail_upload.store(true, Ordering::SeqCst);
    }

    pub(crate) fn downloads(&self) -> usize {
        self.downloads.load(Ordering::SeqCst)
    }
}

impl AccelProvider for RestoreProbeProvider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        "waveform restore probe".into()
    }

    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn upload_numeric_raw(
        &self,
        _: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        if self.fail_upload.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!("late waveform restoration fault").into());
        }
        Ok(self.alias.lock().unwrap().take().expect("configured alias"))
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
}
