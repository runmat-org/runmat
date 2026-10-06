//! Public print-future source/scene ownership through a suspended stage.

use super::*;
use async_trait::async_trait;
use futures::channel::oneshot;
use futures::future::{select, Either};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelIntegerDownloadFuture, AccelNumericDownloadFuture, AccelProvider,
    GpuTensorHandle, GpuTensorStorage, HostNumericDataView, HostNumericTensorView, HostTensorView,
    ThreadProviderGuard,
};
use runmat_filesystem::{
    self as fs, DirEntry, FileHandle, FsMetadata, FsProvider, FsPublicationFailure,
    FsResourceContract, FsResourceLease, FsResourceRevision, FsStagedOutput, OpenFlags,
};
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

struct PausedProvider {
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    started: Mutex<Option<oneshot::Sender<()>>>,
    published: Arc<Mutex<Option<Vec<u8>>>>,
    drops: Arc<AtomicUsize>,
    lease: Arc<RevocableLease>,
}

struct PausedDownloadProvider {
    inner: runmat_accelerate::simple_provider::InProcessProvider,
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    started: Mutex<Option<oneshot::Sender<()>>>,
    downloads: AtomicUsize,
}

impl PausedDownloadProvider {
    async fn wait_for_download(&self) -> anyhow::Result<()> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        let gate = self
            .gate
            .lock()
            .unwrap()
            .take()
            .expect("one resident gather");
        self.started
            .lock()
            .unwrap()
            .take()
            .expect("gather start sender")
            .send(())
            .expect("public test awaits gather start");
        gate.await
            .map_err(|_| anyhow::anyhow!("test download abandoned"))
    }
}

impl AccelProvider for PausedDownloadProvider {
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.wait_for_download().await?;
            self.inner.download_numeric(handle).await
        })
    }

    fn download_integer<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelIntegerDownloadFuture<'a> {
        Box::pin(async move {
            self.wait_for_download().await?;
            self.inner.download_integer(handle).await
        })
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }
}

struct RevocableLease(AtomicBool);

impl RevocableLease {
    fn live() -> Arc<Self> {
        Arc::new(Self(AtomicBool::new(true)))
    }
    fn revoke(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl FsResourceLease for RevocableLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("print.pending.fixture", b"v1"),
            opened_input_handle: None,
            opened_output_handle: None,
        }
    }
    fn is_valid(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

struct PausedStage {
    gate: Option<oneshot::Receiver<()>>,
    bytes: Vec<u8>,
    published: Arc<Mutex<Option<Vec<u8>>>>,
    drops: Arc<AtomicUsize>,
}

impl Drop for PausedStage {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait(?Send)]
impl FsStagedOutput for PausedStage {
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(gate) = self.gate.take() {
            gate.await
                .map_err(|_| io::Error::other("test stage abandoned"))?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    async fn flush_and_sync(&mut self) -> io::Result<()> {
        Ok(())
    }
    async fn publish(mut self: Box<Self>) -> Result<(), FsPublicationFailure> {
        *self.published.lock().unwrap() = Some(std::mem::take(&mut self.bytes));
        Ok(())
    }
}

fn unused() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "unused fixture operation")
}

#[async_trait(?Send)]
impl FsProvider for PausedProvider {
    fn path_namespace(&self) -> fs::FsPathNamespace {
        fs::FsPathNamespace::VirtualRooted
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }
    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(self.lease.clone())
    }
    fn begin_staged_replace(&self, _path: &Path) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        let gate = self
            .gate
            .lock()
            .unwrap()
            .take()
            .expect("one public print stage");
        self.started
            .lock()
            .unwrap()
            .take()
            .expect("stage start sender")
            .send(())
            .expect("public test awaits stage start");
        Ok(Some(Box::new(PausedStage {
            gate: Some(gate),
            bytes: Vec::new(),
            published: self.published.clone(),
            drops: self.drops.clone(),
        })))
    }
    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(unused())
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        Err(unused())
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unused())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unused())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unused())
    }
}

fn await_operation_start<F: Future>(started: oneshot::Receiver<()>, call: Pin<&mut F>) {
    match futures::executor::block_on(select(started, call)) {
        Either::Left((Ok(()), _pending_call)) => {}
        Either::Left((Err(_), _pending_call)) => panic!("selected operation start was abandoned"),
        Either::Right((_finished, _start)) => {
            panic!("public print finished before pending operation")
        }
    }
}

#[test]
fn resident_handle_selects_scene_after_pending_gather_then_retains_it_through_export() {
    let _accelerator = crate::builtins::common::test_support::accel_test_lock();
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::state::reset_hold_state_for_run();
    let _ = super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 3.0, 2.0], vec![3]).unwrap()),
    ]))
    .expect("initial figure");
    let figure = current_figure_handle();
    let original = clone_figure(figure).expect("initial scene");
    let original_png =
        futures::executor::block_on(render_png(original, 800, 600)).expect("initial render");

    let (download_sender, download_receiver) = oneshot::channel();
    let (download_started_sender, download_started_receiver) = oneshot::channel();
    let accelerator = Box::leak(Box::new(PausedDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        gate: Mutex::new(Some(download_receiver)),
        started: Mutex::new(Some(download_started_sender)),
        downloads: AtomicUsize::new(0),
    }));
    let _selected_accelerator = ThreadProviderGuard::set(Some(accelerator));
    let handle = accelerator
        .upload(&HostTensorView {
            data: &[f64::from(figure.as_u32())],
            shape: &[1, 1],
        })
        .expect("real resident scalar handle");

    let (stage_sender, stage_receiver) = oneshot::channel();
    let (stage_started_sender, stage_started_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(stage_receiver)),
        started: Mutex::new(Some(stage_started_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected_filesystem = fs::replace_provider(provider.clone());
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(
        None,
        vec![Value::GpuTensor(handle), Value::from("/resident.png")],
    ));
    await_operation_start(download_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    assert!(
        provider.gate.lock().unwrap().is_some(),
        "no filesystem stage may begin before resident handle materialization"
    );
    let other_filesystem = Arc::new(fs::MemoryFsProvider::with_current_dir("/"));
    let _replacement_filesystem = fs::replace_provider(other_filesystem.clone());

    super::super::state::clear_figure(Some(figure)).expect("mutate selected figure during gather");
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![10.0, 20.0, 30.0], vec![3]).unwrap()),
    ]))
    .expect("replace selected scene");
    let updated = clone_figure(figure).expect("updated scene");
    let updated_png =
        futures::executor::block_on(render_png(updated, 800, 600)).expect("updated render");
    assert_ne!(original_png, updated_png, "fixture must change the scene");

    download_sender.send(()).expect("retained gather receiver");
    await_operation_start(stage_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    super::super::state::clear_figure(Some(figure))
        .expect("clear ambient figure after resolved scene was captured");
    stage_sender.send(()).expect("retained stage receiver");
    assert!(futures::executor::block_on(call).expect("public resident print"));
    assert_eq!(
        published.lock().unwrap().as_deref(),
        Some(updated_png.as_slice())
    );
    assert!(other_filesystem.read_project_path("/resident.png").is_err());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn wide_resident_integer_handle_uses_one_authenticated_download() {
    let _accelerator = crate::builtins::common::test_support::accel_test_lock();
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::state::reset_hold_state_for_run();
    let _ = super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
    ]))
    .expect("selected figure");
    let figure = current_figure_handle();
    let expected = futures::executor::block_on(render_png(
        clone_figure(figure).expect("selected scene"),
        800,
        600,
    ))
    .expect("reference render");

    let (download_sender, download_receiver) = oneshot::channel();
    let (download_started_sender, download_started_receiver) = oneshot::channel();
    let accelerator = Box::leak(Box::new(PausedDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        gate: Mutex::new(Some(download_receiver)),
        started: Mutex::new(Some(download_started_sender)),
        downloads: AtomicUsize::new(0),
    }));
    let _selected_accelerator = ThreadProviderGuard::set(Some(accelerator));
    let number = [u64::from(figure.as_u32())];
    let handle = accelerator
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::U64(&number),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .expect("real U64 resident handle");

    let (stage_sender, stage_receiver) = oneshot::channel();
    let (stage_started_sender, stage_started_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(stage_receiver)),
        started: Mutex::new(Some(stage_started_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected_filesystem = fs::replace_provider(provider.clone());
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(
        None,
        vec![Value::GpuTensor(handle), Value::from("/integer.png")],
    ));
    await_operation_start(download_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    assert!(provider.gate.lock().unwrap().is_some());
    download_sender.send(()).expect("retained integer download");
    await_operation_start(stage_started_receiver, call.as_mut());
    assert_eq!(
        accelerator.downloads.load(Ordering::SeqCst),
        1,
        "the prepared integer payload must not be downloaded again"
    );
    stage_sender.send(()).expect("retained stage receiver");
    assert!(futures::executor::block_on(call).expect("resident integer print"));
    assert_eq!(
        published.lock().unwrap().as_deref(),
        Some(expected.as_slice())
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancelled_pending_resident_gather_has_no_stage_or_download_replay() {
    let _accelerator = crate::builtins::common::test_support::accel_test_lock();
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::state::reset_hold_state_for_run();
    let _ = super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
    ]))
    .expect("selected figure");

    let (download_sender, download_receiver) = oneshot::channel();
    let (download_started_sender, download_started_receiver) = oneshot::channel();
    let accelerator = Box::leak(Box::new(PausedDownloadProvider {
        inner: runmat_accelerate::simple_provider::InProcessProvider::new(),
        gate: Mutex::new(Some(download_receiver)),
        started: Mutex::new(Some(download_started_sender)),
        downloads: AtomicUsize::new(0),
    }));
    let _selected_accelerator = ThreadProviderGuard::set(Some(accelerator));
    let figure_number = [f64::from(current_figure_handle().as_u32())];
    let handle = accelerator
        .upload(&HostTensorView {
            data: &figure_number,
            shape: &[1, 1],
        })
        .expect("real resident scalar handle");

    let (_stage_sender, stage_receiver) = oneshot::channel();
    let (stage_started_sender, _stage_started_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(stage_receiver)),
        started: Mutex::new(Some(stage_started_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected_filesystem = fs::replace_provider(provider.clone());
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(
        Some(runtime),
        vec![
            Value::GpuTensor(handle),
            Value::from("/cancelled-gather.png"),
        ],
    ));
    await_operation_start(download_started_receiver, call.as_mut());
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let error = futures::executor::block_on(call).expect_err("cancelled resident gather");
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    assert!(
        download_sender.send(()).is_err(),
        "download future must be dropped"
    );
    assert_eq!(accelerator.downloads.load(Ordering::SeqCst), 1);
    assert!(provider.gate.lock().unwrap().is_some(), "no stage began");
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}

#[test]
fn public_print_keeps_selected_snapshot_across_pending_stage_and_scene_clear() {
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::state::reset_hold_state_for_run();
    let _ = super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0, 2.0], vec![3]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 3.0, 2.0], vec![3]).unwrap()),
    ]))
    .expect("plot selected scene");
    let original = clone_figure(current_figure_handle()).expect("original scene");
    let expected =
        futures::executor::block_on(render_png(original, 800, 600)).expect("reference render");
    let (sender, receiver) = oneshot::channel();
    let (start_sender, start_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        started: Mutex::new(Some(start_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected = fs::replace_provider(provider.clone());
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(None, vec![Value::from("/selected.png")]));
    await_operation_start(start_receiver, call.as_mut());
    assert!(
        provider.gate.lock().unwrap().is_none(),
        "public call reached selected stage"
    );
    super::super::state::clear_figure(None).expect("clear ambient scene during pending call");
    sender.send(()).expect("retained stage receiver");
    assert!(futures::executor::block_on(call).expect("public print completes"));
    assert_eq!(
        published.lock().unwrap().as_deref(),
        Some(expected.as_slice())
    );
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn cancelled_public_print_drops_pending_stage_without_publication_or_replay() {
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::state::reset_hold_state_for_run();
    let _ = super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
    ]))
    .expect("plot selected scene");
    let (sender, receiver) = oneshot::channel();
    let (start_sender, start_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        started: Mutex::new(Some(start_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected = fs::replace_provider(provider.clone());
    let source = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::with_cancellation(
        std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
        source.token(),
    );
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(
        Some(runtime),
        vec![Value::from("/cancelled.png")],
    ));
    await_operation_start(start_receiver, call.as_mut());
    assert!(
        provider.gate.lock().unwrap().is_none(),
        "public call reached selected stage"
    );
    source.cancel(runmat_execution::CancellationReason::User);
    let error = futures::executor::block_on(call).expect_err("pending print cancelled");
    assert!(error
        .identifier()
        .is_some_and(|id| id.ends_with(":Cancelled")));
    assert!(
        sender.send(()).is_err(),
        "dropped stage must drop pending read owner"
    );
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn revoked_selected_print_lease_after_pending_write_prevents_publication() {
    let _plot = crate::builtins::plotting::tests::lock_plot_registry();
    crate::builtins::plotting::tests::ensure_plot_test_env();
    super::super::state::reset_hold_state_for_run();
    let _ = super::super::state::clear_figure(None);
    let _fs = fs::provider_override_lock();
    futures::executor::block_on(crate::builtins::plotting::plot::plot_builtin(vec![
        Value::Tensor(runmat_value::Tensor::new(vec![0.0, 1.0], vec![2]).unwrap()),
        Value::Tensor(runmat_value::Tensor::new(vec![1.0, 2.0], vec![2]).unwrap()),
    ]))
    .expect("plot selected scene");
    let (sender, receiver) = oneshot::channel();
    let (start_sender, start_receiver) = oneshot::channel();
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let lease = RevocableLease::live();
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        started: Mutex::new(Some(start_sender)),
        published: published.clone(),
        drops: drops.clone(),
        lease: lease.clone(),
    });
    let _selected = fs::replace_provider(provider.clone());
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(None, vec![Value::from("/revoked.png")]));
    await_operation_start(start_receiver, call.as_mut());
    assert!(
        provider.gate.lock().unwrap().is_none(),
        "selected stage owns pending write"
    );
    lease.revoke();
    sender.send(()).expect("resume retained write");
    let error = futures::executor::block_on(call).expect_err("revoked selected lease");
    assert_eq!(error.identifier(), super::PRINT_ERROR_IO.identifier);
    assert!(error.message().contains("revoked"));
    assert!(published.lock().unwrap().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
