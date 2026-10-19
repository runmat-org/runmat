//! The public FID open future retains one provider and drops pending work on
//! invocation cancellation. No provider error is converted into cancellation.

use super::*;
use crate::context::RuntimeContext;
use crate::execution::RuntimeExecutionService;
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_filesystem::{
    DirEntry, FileHandle, FsFilePayloadContract, FsMetadata, FsProvider, FsResourceContract,
    FsResourceLease, FsResourceRevision, MemoryFsProvider, OpenFlags,
};
use runmat_value::Value;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

fn directory() -> &'static str {
    "/fid-await"
}

fn replacement_directory() -> &'static str {
    "/fid-replacement"
}

struct Gate {
    release: AtomicBool,
    entered: AtomicBool,
    dropped: AtomicUsize,
    waker: Mutex<Option<Waker>>,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            release: AtomicBool::new(false),
            entered: AtomicBool::new(false),
            dropped: AtomicUsize::new(0),
            waker: Mutex::new(None),
        })
    }

    fn release(&self) {
        self.release.store(true, Ordering::SeqCst);
        if let Some(waker) = self.waker.lock().expect("gate").take() {
            waker.wake();
        }
    }
}

struct PendingOpen(Arc<Gate>);

impl Drop for PendingOpen {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct Lease(Arc<AtomicBool>);

impl FsResourceLease for Lease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("fid-await-test", b"one"),
            opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            opened_output_handle: None,
        }
    }
    fn is_valid(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

struct Provider {
    memory: MemoryFsProvider,
    gate: Arc<Gate>,
    valid: Arc<AtomicBool>,
}

fn unused() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "unused fixture operation")
}

#[async_trait::async_trait(?Send)]
impl FsProvider for Provider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from(directory()))
    }
    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(Arc::new(Lease(self.valid.clone())))
    }
    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(unused())
    }
    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let _owner = PendingOpen(self.gate.clone());
        futures::future::poll_fn(|cx| {
            self.gate.entered.store(true, Ordering::SeqCst);
            if self.gate.release.load(Ordering::SeqCst) {
                Poll::Ready(())
            } else {
                *self.gate.waker.lock().expect("gate") = Some(cx.waker().clone());
                Poll::Pending
            }
        })
        .await;
        self.memory.open(path, flags)
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.memory.read(path).await
    }
    async fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.memory.write(path, bytes).await
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.metadata(path).await
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unused())
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.memory.canonicalize(path).await
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

fn fixture() -> (Arc<Provider>, Arc<Gate>, Arc<AtomicBool>) {
    let gate = Gate::new();
    let valid = Arc::new(AtomicBool::new(true));
    let memory = MemoryFsProvider::with_current_dir(directory());
    memory
        .write_project_path(Path::new(directory()).join("source.txt"), b"first\n")
        .expect("source fixture");
    (
        Arc::new(Provider {
            memory,
            gate: gate.clone(),
            valid: valid.clone(),
        }),
        gate,
        valid,
    )
}

fn poll_once<F: std::future::Future>(future: std::pin::Pin<&mut F>) -> Poll<F::Output> {
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    future.poll(&mut context)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_open_keeps_original_provider_and_relative_cwd() {
    let _registry = test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    reset_for_tests();
    let (provider, gate, _) = fixture();
    let _selected = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(directory()).expect("selected cwd");
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let args = [Value::from("source.txt"), Value::from("r")];
    let mut call = Box::pin(crate::builtins::io::filetext::fopen::evaluate_in(
        Some(&runtime),
        &args,
    ));
    assert!(poll_once(call.as_mut()).is_pending());
    assert!(gate.entered.load(Ordering::SeqCst));
    let replacement = Arc::new(MemoryFsProvider::with_current_dir(replacement_directory()));
    replacement
        .write_project_path(
            Path::new(replacement_directory()).join("source.txt"),
            b"wrong\n",
        )
        .expect("replacement fixture");
    let _replacement = runmat_filesystem::replace_provider(replacement);
    runmat_filesystem::set_current_dir(replacement_directory()).expect("replacement cwd");
    gate.release();
    let fid = block_on(call)
        .expect("captured open")
        .as_open()
        .expect("open")
        .fid as i32;
    let mut operation = acquire_file_in(fid, Some(&runtime)).expect("session FID");
    let mut bytes = [0u8; 6];
    operation
        .file_mut()
        .read_exact(&mut bytes)
        .expect("selected payload");
    assert_eq!(&bytes, b"first\n");
    drop(operation);
    block_on(close_async_in(fid, Some(&runtime))).expect("close");
    reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_open_cancellation_drops_provider_future_without_registering_fid() {
    let _registry = test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    reset_for_tests();
    let (provider, gate, _) = fixture();
    let _selected = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(directory()).expect("selected cwd");
    let cancellation = CancellationSource::new();
    let runtime = RuntimeContext::with_cancellation(
        Rc::new(RuntimeExecutionService::new()),
        cancellation.token(),
    );
    let args = [Value::from("source.txt"), Value::from("r")];
    let mut call = Box::pin(crate::builtins::io::filetext::fopen::evaluate_in(
        Some(&runtime),
        &args,
    ));
    assert!(poll_once(call.as_mut()).is_pending());
    cancellation.cancel(CancellationReason::User);
    let error = block_on(call).err().expect("cancelled open");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        list_infos_in(Some(&runtime))
            .into_iter()
            .filter(|info| info.id >= 3)
            .count(),
        0
    );
    reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_open_resource_revocation_cannot_publish_fid() {
    let _registry = test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    reset_for_tests();
    let (provider, gate, valid) = fixture();
    let _selected = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(directory()).expect("selected cwd");
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let args = [Value::from("source.txt"), Value::from("r")];
    let mut call = Box::pin(crate::builtins::io::filetext::fopen::evaluate_in(
        Some(&runtime),
        &args,
    ));
    assert!(poll_once(call.as_mut()).is_pending());
    valid.store(false, Ordering::Release);
    gate.release();
    let result = block_on(call).expect("fopen status protocol");
    let opened = result.as_open().expect("open form");
    assert!(opened.fid < 0.0);
    assert_eq!(
        list_infos_in(Some(&runtime))
            .into_iter()
            .filter(|info| info.id >= 3)
            .count(),
        0
    );
    reset_for_tests();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unpolled_and_abandoned_open_never_register_an_identifier() {
    let _registry = test_guard();
    let _filesystem = runmat_filesystem::provider_override_lock();
    reset_for_tests();
    let (provider, gate, _) = fixture();
    let _selected = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir(directory()).expect("selected cwd");
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let args = [Value::from("source.txt"), Value::from("r")];
    drop(crate::builtins::io::filetext::fopen::evaluate_in(
        Some(&runtime),
        &args,
    ));
    assert!(!gate.entered.load(Ordering::SeqCst));
    let mut pending = Box::pin(crate::builtins::io::filetext::fopen::evaluate_in(
        Some(&runtime),
        &args,
    ));
    assert!(poll_once(pending.as_mut()).is_pending());
    drop(pending);
    assert_eq!(gate.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        list_infos_in(Some(&runtime))
            .into_iter()
            .filter(|info| info.id >= 3)
            .count(),
        0
    );
    reset_for_tests();
}
