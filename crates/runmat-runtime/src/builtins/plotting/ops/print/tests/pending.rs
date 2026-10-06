//! Public print-future source/scene ownership through a suspended stage.

use super::*;
use async_trait::async_trait;
use futures::channel::oneshot;
use futures::task::noop_waker;
use runmat_filesystem::{
    self as fs, DirEntry, FileHandle, FsMetadata, FsProvider, FsPublicationFailure,
    FsResourceContract, FsResourceLease, FsResourceRevision, FsStagedOutput, OpenFlags,
};
use std::future::Future;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

struct PausedProvider {
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    published: Arc<Mutex<Option<Vec<u8>>>>,
    drops: Arc<AtomicUsize>,
    lease: Arc<RevocableLease>,
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
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        published: published.clone(),
        drops: drops.clone(),
        lease: RevocableLease::live(),
    });
    let _selected = fs::replace_provider(provider.clone());
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(None, vec![Value::from("/selected.png")]));
    let waker = noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
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
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
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
    let waker = noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
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
    let published = Arc::new(Mutex::new(None));
    let drops = Arc::new(AtomicUsize::new(0));
    let lease = RevocableLease::live();
    let provider = Arc::new(PausedProvider {
        gate: Mutex::new(Some(receiver)),
        published: published.clone(),
        drops: drops.clone(),
        lease: lease.clone(),
    });
    let _selected = fs::replace_provider(provider.clone());
    let _statement = crate::output_count::push_output_count(Some(0));
    let mut call = Box::pin(print_builtin(None, vec![Value::from("/revoked.png")]));
    let waker = noop_waker();
    assert!(matches!(
        call.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
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
