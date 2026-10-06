use super::*;
use async_trait::async_trait;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsPathNamespace, FsProvider, OpenFlags};
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(not(feature = "plot-core"))]
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostLogicalTensorView, HostNumericTensorView, HostTensorView, ProviderPrecision,
    ThreadProviderGuard,
};

struct CountingFilesystem {
    inner: runmat_filesystem::MemoryFsProvider,
    path_captures: AtomicUsize,
}

impl CountingFilesystem {
    fn new() -> Self {
        Self {
            inner: runmat_filesystem::MemoryFsProvider::with_current_dir("/print"),
            path_captures: AtomicUsize::new(0),
        }
    }

    fn capture_count(&self) -> usize {
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
struct CountingDownloadProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    downloads: AtomicUsize,
}

#[cfg(not(feature = "plot-core"))]
impl AccelProvider for CountingDownloadProvider {
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }

    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> anyhow::Result<GpuTensorHandle> {
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

fn runtime(mode: crate::context::RuntimeLanguageMode) -> crate::context::RuntimeContext {
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    runtime.set_language_mode(mode);
    runtime
        .set_runmat_extensions_enabled(matches!(mode, crate::context::RuntimeLanguageMode::RunMat));
    runtime
}

async fn dispatch(
    runtime: &crate::context::RuntimeContext,
    path: &str,
    outputs: usize,
) -> Result<Value, RuntimeError> {
    crate::call_builtin_async_with_outputs_in_context(
        runtime,
        "print",
        &[Value::from(path), Value::from("-dpng")],
        outputs,
    )
    .await
}

fn assert_no_outputs(value: Value) {
    assert!(matches!(value, Value::OutputList(values) if values.is_empty()));
}

#[cfg(feature = "plot-core")]
#[test]
fn dispatcher_uses_selected_compatibility_and_requested_output_count() {
    let _plot = setup();
    let _filesystem_lock = runmat_filesystem::provider_override_lock();
    block_on(plot_builtin(vec![
        Value::Tensor(tensor(&[0.0, 1.0, 2.0])),
        Value::Tensor(tensor(&[2.0, 1.0, 3.0])),
    ]))
    .expect("plot fixture");
    let filesystem = Arc::new(CountingFilesystem::new());
    let _filesystem = runmat_filesystem::replace_provider(filesystem.clone());
    let matlab = runtime(crate::context::RuntimeLanguageMode::Matlab);
    let runmat = runtime(crate::context::RuntimeLanguageMode::RunMat);

    let ambient_runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    assert_no_outputs(block_on(dispatch(&matlab, "/print/matlab-statement.png", 0)).unwrap());
    let error = block_on(dispatch(&matlab, "/print/matlab-status.png", 1)).unwrap_err();
    assert_eq!(
        error.identifier(),
        PRINT_STATUS_OUTPUT_EXTENSION.error_identifier
    );
    let error = block_on(dispatch(&matlab, "/print/matlab-two.png", 2)).unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(ambient_runmat);

    let ambient_matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    assert_no_outputs(block_on(dispatch(&runmat, "/print/runmat-statement.png", 0)).unwrap());
    assert_eq!(
        block_on(dispatch(&runmat, "/print/runmat-status.png", 1)).unwrap(),
        Value::Bool(true)
    );
    let error = block_on(dispatch(&runmat, "/print/runmat-two.png", 2)).unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(ambient_matlab);

    assert!(filesystem
        .inner
        .read_project_path("/print/matlab-statement.png")
        .is_ok());
    assert!(filesystem
        .inner
        .read_project_path("/print/runmat-statement.png")
        .is_ok());
    assert!(filesystem
        .inner
        .read_project_path("/print/runmat-status.png")
        .is_ok());
    assert!(filesystem
        .inner
        .read_project_path("/print/matlab-status.png")
        .is_err());
    assert!(filesystem.capture_count() > 0);
}

#[cfg(not(feature = "plot-core"))]
fn assert_renderer_unavailable(error: &RuntimeError) {
    assert_eq!(error.identifier(), PRINT_ERROR_RENDER.identifier);
    let mut source = std::error::Error::source(error);
    let mut found = false;
    while let Some(error) = source {
        found |= error
            .downcast_ref::<crate::builtins::plotting::engine::PngExportUnavailable>()
            .is_some();
        source = error.source();
    }
    assert!(
        found,
        "typed PNG renderer capability cause must be retained"
    );
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn feature_disabled_dispatch_fails_before_filesystem_capture() {
    let _plot = setup();
    let _filesystem_lock = runmat_filesystem::provider_override_lock();
    let filesystem = Arc::new(CountingFilesystem::new());
    let _filesystem = runmat_filesystem::replace_provider(filesystem.clone());
    let matlab = runtime(crate::context::RuntimeLanguageMode::Matlab);
    let runmat = runtime(crate::context::RuntimeLanguageMode::RunMat);

    let ambient_runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    assert_renderer_unavailable(
        &block_on(dispatch(&matlab, "/print/matlab-statement.png", 0)).unwrap_err(),
    );
    let error = block_on(dispatch(&matlab, "/print/matlab-status.png", 1)).unwrap_err();
    assert_eq!(
        error.identifier(),
        PRINT_STATUS_OUTPUT_EXTENSION.error_identifier
    );
    let error = block_on(dispatch(&matlab, "/print/matlab-two.png", 2)).unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(ambient_runmat);

    let ambient_matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    assert_renderer_unavailable(
        &block_on(dispatch(&runmat, "/print/runmat-statement.png", 0)).unwrap_err(),
    );
    assert_renderer_unavailable(
        &block_on(dispatch(&runmat, "/print/runmat-status.png", 1)).unwrap_err(),
    );
    let error = block_on(dispatch(&runmat, "/print/runmat-two.png", 2)).unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_TOO_MANY_OUTPUTS.identifier);
    drop(ambient_matlab);

    assert_eq!(filesystem.capture_count(), 0);
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn known_invalid_host_argument_precedes_renderer_capability() {
    let _plot = setup();
    let _filesystem_lock = runmat_filesystem::provider_override_lock();
    let filesystem = Arc::new(CountingFilesystem::new());
    let _filesystem = runmat_filesystem::replace_provider(filesystem.clone());
    let runtime = runtime(crate::context::RuntimeLanguageMode::RunMat);
    let nested = runmat_value::CellArray::new(vec![Value::from("plot.png")], 1, 1)
        .expect("one-cell fixture");
    let error = block_on(crate::call_builtin_async_with_outputs_in_context(
        &runtime,
        "print",
        &[Value::Cell(nested)],
        0,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), PRINT_ERROR_INVALID_INPUT.identifier);
    assert_eq!(filesystem.capture_count(), 0);
}

#[cfg(not(feature = "plot-core"))]
#[test]
fn feature_disabled_dispatch_does_not_download_resident_handle() {
    let _plot = setup();
    let _acceleration = crate::builtins::common::test_support::accel_test_lock();
    let provider = Box::leak(Box::new(CountingDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        downloads: AtomicUsize::new(0),
    }));
    let _provider = ThreadProviderGuard::set(Some(provider));
    let handle = current_figure_handle();
    let host = Tensor::new(vec![f64::from(handle.as_u32())], vec![1, 1]).unwrap();
    let resident = crate::builtins::common::gpu_helpers::upload_tensor(provider, &host).unwrap();
    let runtime = runtime(crate::context::RuntimeLanguageMode::RunMat);
    let error = block_on(crate::call_builtin_async_with_outputs_in_context(
        &runtime,
        "print",
        &[
            Value::GpuTensor(resident),
            Value::from("/never-captured.png"),
        ],
        0,
    ))
    .unwrap_err();
    assert_renderer_unavailable(&error);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
}
