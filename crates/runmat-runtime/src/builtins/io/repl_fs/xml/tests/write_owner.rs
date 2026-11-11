//! Public write invocation witnesses across a real execution-service turn.

use super::*;
use crate::context::RuntimeServiceFuture;
use crate::execution::RootedValueSequence;
use crate::execution::{
    AwaitAction, CpuCooperationPolicy, DeferredCall, ExecutionServiceError,
    RuntimeExecutionServices,
};
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelNumericDownloadFuture, AccelProvider, GpuTensorHandle,
    HostTensorView, ProviderPrecision,
};
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::{
    CancellationReason, ExecutionHandleSnapshot, ExecutionScopeId, FutureHandle, PoolHandle,
    PoolRequest, PoolSnapshot, TaskHandle, TaskResultClaim,
};
use std::cell::{Cell, RefCell};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

struct Turn {
    release: Cell<bool>,
    entered: Cell<bool>,
    waker: RefCell<Option<Waker>>,
}

impl Turn {
    fn new() -> Rc<Self> {
        Rc::new(Self {
            release: Cell::new(false),
            entered: Cell::new(false),
            waker: RefCell::new(None),
        })
    }

    fn release(&self) {
        self.release.set(true);
        if let Some(waker) = self.waker.borrow_mut().take() {
            waker.wake();
        }
    }
}

struct TurnService(Rc<Turn>);

impl RuntimeExecutionServices for TurnService {
    fn cpu_cooperation_policy(&self) -> CpuCooperationPolicy {
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("nonzero"))
    }

    fn cooperate_cpu(&self) -> RuntimeServiceFuture<Result<(), ExecutionServiceError>> {
        let turn = self.0.clone();
        Box::pin(futures::future::poll_fn(move |cx| {
            turn.entered.set(true);
            if turn.release.get() {
                Poll::Ready(Ok(()))
            } else {
                *turn.waker.borrow_mut() = Some(cx.waker().clone());
                Poll::Pending
            }
        }))
    }

    fn scope_id(&self) -> ExecutionScopeId {
        ExecutionScopeId::derive(&[b"xml-write-owner-test"])
    }
    fn current_pool(&self) -> Result<Option<PoolSnapshot>, ExecutionServiceError> {
        Ok(None)
    }
    fn ensure_pool(&self, _: PoolRequest) -> Result<PoolSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn close_pool(&self, _: &PoolHandle) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn create_future(&self, _: DeferredCall) -> Result<FutureHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn spawn(&self, _: &FutureHandle) -> Result<TaskHandle, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn inspect_future(
        &self,
        _: &FutureHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn inspect_task(
        &self,
        _: &TaskHandle,
    ) -> Result<ExecutionHandleSnapshot, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn claim_next_task_result(
        &self,
        _: &[runmat_value::SharedTaskHandle],
    ) -> Result<TaskResultClaim, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn mark_task_result_read(&self, _: &TaskHandle) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn begin_await(&self, _: Value) -> Result<AwaitAction, ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn complete_future(
        &self,
        _: crate::execution::FutureCompletionRef<'_>,
        _: Result<RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn cancel(&self, _: &Value, _: CancellationReason) -> Result<(), ExecutionServiceError> {
        Err(ExecutionServiceError::UnknownHandle)
    }
    fn drain_scope(&self, _: CancellationReason) {}
}

fn poll_once<F: std::future::Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    future.poll(&mut context)
}

fn selected_dir() -> &'static str {
    "/selected"
}

fn replacement_dir() -> &'static str {
    "/replacement"
}

fn selected_path(name: &str) -> PathBuf {
    Path::new(selected_dir()).join(name)
}

fn replacement_path(name: &str) -> PathBuf {
    Path::new(replacement_dir()).join(name)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn suspended_write_keeps_original_provider_and_cwd() {
    let _guard = fs_test_guard();
    let selected = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        selected_dir(),
    ));
    let replacement = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        replacement_dir(),
    ));
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let turn = Turn::new();
    let runtime = RuntimeContext::new(Rc::new(TurnService(turn.clone())));
    let mut call = Box::pin(xmlwrite_builtin(
        Some(runtime),
        vec![Value::from("written.xml"), Value::from("<root/>")],
    ));
    assert!(poll_once(call.as_mut()).is_pending());
    assert!(
        turn.entered.get(),
        "write must reach an actual pending turn"
    );
    let _replacement = runmat_filesystem::replace_provider(replacement.clone());
    runmat_filesystem::set_current_dir(replacement_dir()).expect("replacement cwd");
    turn.release();
    futures::executor::block_on(call).expect("captured-owner write");
    assert_eq!(
        futures::executor::block_on(selected.read(&selected_path("written.xml")))
            .expect("selected output"),
        b"<root/>"
    );
    assert!(
        futures::executor::block_on(replacement.read(&replacement_path("written.xml"))).is_err()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancelled_pending_write_does_not_publish() {
    let _guard = fs_test_guard();
    let selected = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        selected_dir(),
    ));
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let turn = Turn::new();
    let cancellation = CancellationSource::new();
    let runtime =
        RuntimeContext::with_cancellation(Rc::new(TurnService(turn.clone())), cancellation.token());
    let mut call = Box::pin(xmlwrite_builtin(
        Some(runtime),
        vec![Value::from("cancelled.xml"), Value::from("<root/>")],
    ));
    assert!(poll_once(call.as_mut()).is_pending());
    assert!(turn.entered.get());
    cancellation.cancel(CancellationReason::User);
    turn.release();
    let error = futures::executor::block_on(call).expect_err("cancelled write");
    assert_eq!(error.identifier(), Some("RunMat:xmlwrite:Cancelled"));
    assert!(futures::executor::block_on(selected.read(&selected_path("cancelled.xml"))).is_err());
}

struct RevocableLease(Arc<AtomicBool>);

impl runmat_filesystem::FsResourceLease for RevocableLease {
    fn contract(&self) -> runmat_filesystem::FsResourceContract {
        runmat_filesystem::FsResourceContract {
            revision: runmat_filesystem::FsResourceRevision::for_provider(
                "xml-write-test",
                b"revocable-v1",
            ),
            opened_input_handle: None,
            opened_output_handle: None,
        }
    }

    fn is_valid(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

struct RevocableProvider {
    memory: runmat_filesystem::MemoryFsProvider,
    valid: Arc<AtomicBool>,
    opens: Arc<AtomicUsize>,
    reads: Arc<AtomicUsize>,
    writes: Arc<AtomicUsize>,
}

#[path = "write_owner/pre_revoked.rs"]
mod pre_revoked;

fn unsupported() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Unsupported, "unused fixture operation")
}

#[async_trait::async_trait(?Send)]
impl runmat_filesystem::FsProvider for RevocableProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn resource_lease(&self) -> Option<Arc<dyn runmat_filesystem::FsResourceLease>> {
        Some(Arc::new(RevocableLease(self.valid.clone())))
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from(selected_dir()))
    }
    fn open(
        &self,
        _: &Path,
        _: &runmat_filesystem::OpenFlags,
    ) -> std::io::Result<Box<dyn runmat_filesystem::FileHandle>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        Err(unsupported())
    }
    async fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.memory.read(path).await
    }
    async fn write(&self, path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.memory.write(path, bytes).await
    }
    async fn remove_file(&self, _: &Path) -> std::io::Result<()> {
        Err(unsupported())
    }
    async fn metadata(&self, path: &Path) -> std::io::Result<runmat_filesystem::FsMetadata> {
        self.memory.metadata(path).await
    }
    async fn symlink_metadata(&self, _: &Path) -> std::io::Result<runmat_filesystem::FsMetadata> {
        Err(unsupported())
    }
    async fn read_dir(&self, _: &Path) -> std::io::Result<Vec<runmat_filesystem::DirEntry>> {
        Err(unsupported())
    }
    async fn canonicalize(&self, path: &Path) -> std::io::Result<PathBuf> {
        self.memory.canonicalize(path).await
    }
    async fn create_dir(&self, _: &Path) -> std::io::Result<()> {
        Err(unsupported())
    }
    async fn create_dir_all(&self, _: &Path) -> std::io::Result<()> {
        Err(unsupported())
    }
    async fn remove_dir(&self, _: &Path) -> std::io::Result<()> {
        Err(unsupported())
    }
    async fn remove_dir_all(&self, _: &Path) -> std::io::Result<()> {
        Err(unsupported())
    }
    async fn rename(&self, _: &Path, _: &Path) -> std::io::Result<()> {
        Err(unsupported())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> std::io::Result<()> {
        Err(unsupported())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn revoked_during_turn_rejects_write_before_provider_effect() {
    let _guard = fs_test_guard();
    let valid = Arc::new(AtomicBool::new(true));
    let writes = Arc::new(AtomicUsize::new(0));
    let selected = Arc::new(RevocableProvider {
        memory: runmat_filesystem::MemoryFsProvider::with_current_dir(selected_dir()),
        valid: valid.clone(),
        opens: Arc::new(AtomicUsize::new(0)),
        reads: Arc::new(AtomicUsize::new(0)),
        writes: writes.clone(),
    });
    let _selected = runmat_filesystem::replace_provider(selected.clone());
    runmat_filesystem::set_current_dir(selected_dir()).expect("selected cwd");
    let turn = Turn::new();
    let runtime = RuntimeContext::new(Rc::new(TurnService(turn.clone())));
    let mut call = Box::pin(xmlwrite_builtin(
        Some(runtime),
        vec![Value::from("revoked.xml"), Value::from("<root/>")],
    ));
    assert!(poll_once(call.as_mut()).is_pending());
    assert!(turn.entered.get());
    valid.store(false, Ordering::Release);
    turn.release();
    let error = futures::executor::block_on(call).expect_err("revoked owner");
    assert_eq!(error.identifier(), Some("RunMat:xmlwrite:IoFailed"));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(
        writes.load(Ordering::SeqCst),
        0,
        "no provider write after revoke"
    );
    assert!(futures::executor::block_on(selected.read(&selected_path("revoked.xml"))).is_err());
}
