use std::future::Future;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use crate::context::{RuntimeHttpService, RuntimeServiceFuture};
use crate::http::{HttpRequest, HttpResponse, TransportError};
use async_trait::async_trait;
use runmat_execution::cancellation::CancellationSource;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, FsResourceContract, FsResourceLease,
    FsResourceRevision, MemoryFsProvider, OpenFlags,
};

struct PendingRead {
    dropped: Arc<AtomicUsize>,
}

struct PendingWrite {
    dropped: Arc<AtomicUsize>,
}

impl Future for PendingWrite {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingWrite {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl Future for PendingRead {
    type Output = io::Result<Vec<u8>>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingRead {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct RevocableLease {
    valid: Arc<AtomicBool>,
}

impl FsResourceLease for RevocableLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("archive-owner-test", b"v1"),
            opened_input_handle: None,
            opened_output_handle: None,
        }
    }

    fn is_valid(&self) -> bool {
        self.valid.load(Ordering::SeqCst)
    }
}

struct AmbientReplacement {
    provider: Arc<dyn FsProvider>,
    restoration: Arc<Mutex<Option<runmat_filesystem::ProviderGuard>>>,
}

pub(super) struct AmbientSwapRestoration<'a> {
    _selected: &'a runmat_filesystem::ProviderGuard,
    restoration: Arc<Mutex<Option<runmat_filesystem::ProviderGuard>>>,
}

impl<'a> AmbientSwapRestoration<'a> {
    pub(super) fn new(
        selected: &'a runmat_filesystem::ProviderGuard,
        restoration: Arc<Mutex<Option<runmat_filesystem::ProviderGuard>>>,
    ) -> Self {
        Self {
            _selected: selected,
            restoration,
        }
    }
}

impl Drop for AmbientSwapRestoration<'_> {
    fn drop(&mut self) {
        // Restore the callback's nested swap before the selected scope, even
        // when an assertion unwinds. Release the cell lock before its guard.
        let nested = self
            .restoration
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        drop(nested);
    }
}

pub(super) struct RecordingProvider {
    pub(super) memory: MemoryFsProvider,
    pub(super) valid: Arc<AtomicBool>,
    pub(super) reads: Arc<AtomicUsize>,
    pub(super) writes: Arc<AtomicUsize>,
    pub(super) creates: Arc<AtomicUsize>,
    pending_reads: bool,
    pub(super) pending_dropped: Arc<AtomicUsize>,
    revoke_after_read: bool,
    read_error: Option<io::ErrorKind>,
    pending_writes: bool,
    pub(super) pending_write_dropped: Arc<AtomicUsize>,
    revoke_after_read_dir: bool,
    pub(super) read_dirs: Arc<AtomicUsize>,
    pub(super) metadata_calls: Arc<AtomicUsize>,
    pub(super) symlink_metadata_calls: Arc<AtomicUsize>,
    metadata_failure: Option<(usize, io::ErrorKind)>,
    replacement_on_read: Option<AmbientReplacement>,
}

impl RecordingProvider {
    pub(super) fn new(cwd: &str) -> Self {
        Self {
            memory: MemoryFsProvider::with_current_dir(cwd),
            valid: Arc::new(AtomicBool::new(true)),
            reads: Arc::new(AtomicUsize::new(0)),
            writes: Arc::new(AtomicUsize::new(0)),
            creates: Arc::new(AtomicUsize::new(0)),
            pending_reads: false,
            pending_dropped: Arc::new(AtomicUsize::new(0)),
            revoke_after_read: false,
            read_error: None,
            pending_writes: false,
            pending_write_dropped: Arc::new(AtomicUsize::new(0)),
            revoke_after_read_dir: false,
            read_dirs: Arc::new(AtomicUsize::new(0)),
            metadata_calls: Arc::new(AtomicUsize::new(0)),
            symlink_metadata_calls: Arc::new(AtomicUsize::new(0)),
            metadata_failure: None,
            replacement_on_read: None,
        }
    }

    pub(super) fn with_pending_reads(mut self) -> Self {
        self.pending_reads = true;
        self
    }

    pub(super) fn with_read_revocation(mut self) -> Self {
        self.revoke_after_read = true;
        self
    }

    pub(super) fn with_read_error_and_revocation(mut self, kind: io::ErrorKind) -> Self {
        self.read_error = Some(kind);
        self.revoke_after_read = true;
        self
    }

    pub(super) fn with_pending_writes(mut self) -> Self {
        self.pending_writes = true;
        self
    }

    pub(super) fn with_read_dir_revocation(mut self) -> Self {
        self.revoke_after_read_dir = true;
        self
    }

    pub(super) fn with_metadata_error_and_revocation(
        mut self,
        call: usize,
        kind: io::ErrorKind,
    ) -> Self {
        self.metadata_failure = Some((call, kind));
        self
    }

    pub(super) fn with_replacement_on_read(
        mut self,
        replacement: Arc<dyn FsProvider>,
        guard: Arc<Mutex<Option<runmat_filesystem::ProviderGuard>>>,
    ) -> Self {
        self.replacement_on_read = Some(AmbientReplacement {
            provider: replacement,
            restoration: guard,
        });
        self
    }
}

#[async_trait(?Send)]
impl FsProvider for RecordingProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(Arc::new(RevocableLease {
            valid: Arc::clone(&self.valid),
        }))
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        self.memory.current_dir_override()
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.memory.open(path, flags)
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if let Some(replacement) = &self.replacement_on_read {
            let mut guard = replacement
                .restoration
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if guard.is_none() {
                *guard = Some(runmat_filesystem::replace_provider(Arc::clone(
                    &replacement.provider,
                )));
            }
        }
        if self.pending_reads {
            return PendingRead {
                dropped: Arc::clone(&self.pending_dropped),
            }
            .await;
        }
        let result = match self.read_error {
            Some(kind) => Err(io::Error::new(kind, "selected provider read failed")),
            None => self.memory.read(path).await,
        };
        if self.revoke_after_read {
            self.valid.store(false, Ordering::SeqCst);
        }
        result
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        if self.pending_writes {
            return PendingWrite {
                dropped: Arc::clone(&self.pending_write_dropped),
            }
            .await;
        }
        self.memory.write(path, data).await
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_file(path).await
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let call = self.metadata_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some((failure_at, kind)) = self.metadata_failure {
            if call == failure_at {
                self.valid.store(false, Ordering::SeqCst);
                return Err(io::Error::new(kind, "selected provider metadata failed"));
            }
        }
        self.memory.metadata(path).await
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.symlink_metadata_calls.fetch_add(1, Ordering::SeqCst);
        self.memory.symlink_metadata(path).await
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.read_dirs.fetch_add(1, Ordering::SeqCst);
        let result = self.memory.read_dir(path).await;
        if result.is_ok() && self.revoke_after_read_dir {
            self.valid.store(false, Ordering::SeqCst);
        }
        result
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.memory.canonicalize(path).await
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir(path).await
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.creates.fetch_add(1, Ordering::SeqCst);
        self.memory.create_dir_all(path).await
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir(path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir_all(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.memory.rename(from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.memory.set_readonly(path, readonly).await
    }
}

pub(super) fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    future.poll(&mut context)
}

pub(super) fn runtime(cancellation: &CancellationSource) -> crate::context::RuntimeContext {
    crate::context::RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
        .with_cancellation_token(cancellation.token())
}

pub(super) fn encoded(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(bytes).expect("encode fixture");
    encoder.finish().expect("finish fixture")
}

pub(super) struct ReplacingHttpService {
    pub(super) body: Vec<u8>,
    pub(super) sends: Arc<AtomicUsize>,
    pub(super) replacement: Arc<dyn FsProvider>,
    pub(super) guard: Arc<Mutex<Option<runmat_filesystem::ProviderGuard>>>,
}

impl RuntimeHttpService for ReplacingHttpService {
    fn send(
        &self,
        _: crate::context::RuntimeContext,
        _: HttpRequest,
    ) -> RuntimeServiceFuture<Result<HttpResponse, TransportError>> {
        self.sends.fetch_add(1, Ordering::SeqCst);
        let mut guard = self
            .guard
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if guard.is_none() {
            *guard = Some(runmat_filesystem::replace_provider(Arc::clone(
                &self.replacement,
            )));
        }
        let body = self.body.clone();
        Box::pin(async move {
            Ok(HttpResponse {
                headers: Vec::new(),
                body,
            })
        })
    }
}
