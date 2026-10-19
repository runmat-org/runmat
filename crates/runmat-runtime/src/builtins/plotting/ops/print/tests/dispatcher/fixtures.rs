use super::super::*;
use async_trait::async_trait;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsPathNamespace, FsProvider, OpenFlags};
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(feature = "plot-core"))]
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostLogicalTensorView, HostNumericTensorView, HostTensorView, ProviderPrecision,
};

pub(super) struct CountingFilesystem {
    pub(super) inner: runmat_filesystem::MemoryFsProvider,
    path_captures: AtomicUsize,
}

impl CountingFilesystem {
    pub(super) fn new() -> Self {
        Self {
            inner: runmat_filesystem::MemoryFsProvider::with_current_dir("/print"),
            path_captures: AtomicUsize::new(0),
        }
    }

    pub(super) fn capture_count(&self) -> usize {
        self.path_captures.load(Ordering::SeqCst)
    }
}

#[async_trait(?Send)]
impl FsProvider for CountingFilesystem {
    fn path_namespace(&self) -> FsPathNamespace {
        self.path_captures.fetch_add(1, Ordering::SeqCst);
        self.inner.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        self.inner.current_dir_override()
    }

    fn begin_staged_replace(
        &self,
        destination: &Path,
    ) -> io::Result<Option<Box<dyn runmat_filesystem::FsStagedOutput>>> {
        self.inner.begin_staged_replace(destination)
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.inner.open(path, flags)
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.inner.read(path).await
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.inner.write(path, data).await
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_file(path).await
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.inner.metadata(path).await
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.inner.symlink_metadata(path).await
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.inner.read_dir(path).await
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.inner.canonicalize(path).await
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir(path).await
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir_all(path).await
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir(path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir_all(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.inner.rename(from, to).await
    }

    async fn rename_replace(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.inner.rename_replace(from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.inner.set_readonly(path, readonly).await
    }
}

#[cfg(not(feature = "plot-core"))]
pub(super) struct CountingDownloadProvider {
    pub(super) inner: runmat_accelerate::simple_provider::InProcessProvider,
    pub(super) downloads: AtomicUsize,
}

#[cfg(not(feature = "plot-core"))]
impl AccelProvider for CountingDownloadProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }

    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_logical_raw(host)
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

pub(super) fn runtime(mode: crate::context::RuntimeLanguageMode) -> crate::context::RuntimeContext {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    runtime.set_language_mode(mode);
    runtime
        .set_runmat_extensions_enabled(matches!(mode, crate::context::RuntimeLanguageMode::RunMat));
    runtime
}

pub(super) async fn dispatch(
    runtime: &crate::context::RuntimeContext,
    path: &str,
    outputs: usize,
) -> Result<crate::execution::RootedValue, RuntimeError> {
    crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "print",
        &[Value::from(path), Value::from("-dpng")],
        outputs,
    )
    .await
}

pub(super) fn assert_no_outputs(value: crate::execution::RootedValue) {
    assert!(matches!(value.value(), Value::OutputList(values) if values.is_empty()));
}
