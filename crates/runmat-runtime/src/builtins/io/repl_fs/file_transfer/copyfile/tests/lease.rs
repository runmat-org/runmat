use async_trait::async_trait;
use runmat_filesystem::{
    DirEntry, FileHandle, FsFilePayloadContract, FsMetadata, FsProvider, FsResourceContract,
    FsResourceLease, FsResourceRevision, MemoryFsProvider, OpenFlags,
};
use std::future::Future;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

struct RevocableLease {
    valid: Arc<AtomicBool>,
}

impl FsResourceLease for RevocableLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("copyfile-lease-test", b"v1"),
            opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            opened_output_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
        }
    }

    fn is_valid(&self) -> bool {
        self.valid.load(Ordering::SeqCst)
    }
}

struct RevokeOnWrite {
    inner: Box<dyn FileHandle>,
    valid: Arc<AtomicBool>,
    writes: Arc<AtomicUsize>,
}

impl Read for RevokeOnWrite {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.inner.read(bytes)
    }
}

impl Write for RevokeOnWrite {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.valid.store(false, Ordering::SeqCst);
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for RevokeOnWrite {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.inner.seek(from)
    }
}

#[async_trait(?Send)]
impl FileHandle for RevokeOnWrite {}

struct RevokingProvider {
    memory: MemoryFsProvider,
    valid: Arc<AtomicBool>,
    writes: Arc<AtomicUsize>,
    pending_rename_dropped: Option<Arc<AtomicBool>>,
}

struct PendingRename(Arc<AtomicBool>);

impl Future for PendingRename {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingRename {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[async_trait(?Send)]
impl FsProvider for RevokingProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(Arc::new(RevocableLease {
            valid: self.valid.clone(),
        }))
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/selected"))
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let inner = self.memory.open(path, flags)?;
        if path == Path::new("/selected/target.bin") && flags.write {
            Ok(Box::new(RevokeOnWrite {
                inner,
                valid: self.valid.clone(),
                writes: self.writes.clone(),
            }))
        } else {
            Ok(inner)
        }
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.memory.read(path).await
    }
    async fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.memory.write(path, bytes).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_file(path).await
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.metadata(path).await
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.symlink_metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.memory.read_dir(path).await
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.memory.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        if let Some(dropped) = &self.pending_rename_dropped {
            return PendingRename(dropped.clone()).await;
        }
        self.memory.rename(from, to).await?;
        self.valid.store(false, Ordering::SeqCst);
        Ok(())
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.memory.set_readonly(path, readonly).await
    }
}

#[test]
fn revoked_lease_after_first_chunk_prevents_next_provider_effect() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let memory = MemoryFsProvider::with_current_dir("/selected");
    memory
        .write_project_path("/selected/source.bin", &vec![9u8; 32 * 1024])
        .expect("source");
    let writes = Arc::new(AtomicUsize::new(0));
    let provider = Arc::new(RevokingProvider {
        memory: memory.clone(),
        valid: Arc::new(AtomicBool::new(true)),
        writes: writes.clone(),
        pending_rename_dropped: None,
    });
    let captured = runmat_filesystem::with_provider_override(provider, || {
        super::super::TransferFilesystem::capture(
            super::super::super::authority::TransferOperation::Copy,
            None,
        )
        .expect("capture lease")
    });
    let outcome = futures::executor::block_on(super::super::single::copy(
        &captured,
        "source.bin",
        "target.bin",
        false,
    ));
    assert_eq!(outcome.status(), 0.0);
    assert_eq!(writes.load(Ordering::SeqCst), 1);
    assert_eq!(
        memory
            .read_project_path("/selected/target.bin")
            .unwrap()
            .len(),
        16 * 1024
    );
}

#[test]
fn move_reports_resource_lease_retirement_after_provider_rename() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let memory = MemoryFsProvider::with_current_dir("/selected");
    memory
        .write_project_path("/selected/source.bin", b"source")
        .expect("source");
    let provider = Arc::new(RevokingProvider {
        memory: memory.clone(),
        valid: Arc::new(AtomicBool::new(true)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_rename_dropped: None,
    });
    let captured = runmat_filesystem::with_provider_override(provider, || {
        super::super::super::authority::TransferFilesystem::capture(
            super::super::super::authority::TransferOperation::Move,
            None,
        )
        .expect("capture move lease")
    });

    let outcome = futures::executor::block_on(super::super::super::movefile::single::move_path(
        &captured,
        "source.bin",
        "target.bin",
        false,
    ))
    .expect("lease retirement is a filesystem status result");

    assert_eq!(outcome.status(), 0.0);
    assert!(memory.read_project_path("/selected/source.bin").is_err());
    assert_eq!(
        memory.read_project_path("/selected/target.bin").unwrap(),
        b"source"
    );
}

#[test]
fn selected_move_builtin_drops_pending_provider_without_replay() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let memory = MemoryFsProvider::with_current_dir("/selected");
    memory
        .write_project_path("/selected/source.bin", b"source")
        .expect("source");
    let pending_rename_dropped = Arc::new(AtomicBool::new(false));
    let provider = Arc::new(RevokingProvider {
        memory: memory.clone(),
        valid: Arc::new(AtomicBool::new(true)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_rename_dropped: Some(pending_rename_dropped.clone()),
    });
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    runmat_filesystem::with_provider_override(provider, || {
        let mut call = Box::pin(super::super::super::movefile::movefile_builtin(
            Some(runtime),
            vec![
                runmat_value::Value::from("source.bin"),
                runmat_value::Value::from("target.bin"),
            ],
        ));
        assert!(matches!(
            call.as_mut()
                .poll(&mut Context::from_waker(futures::task::noop_waker_ref())),
            Poll::Pending
        ));
        drop(call);
    });

    assert!(pending_rename_dropped.load(Ordering::SeqCst));
    assert_eq!(
        memory.read_project_path("/selected/source.bin").unwrap(),
        b"source"
    );
    assert!(memory.read_project_path("/selected/target.bin").is_err());
}

#[test]
fn selected_move_builtin_cancels_pending_provider_with_typed_error() {
    let _lock = crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let memory = MemoryFsProvider::with_current_dir("/selected");
    memory
        .write_project_path("/selected/source.bin", b"source")
        .expect("source");
    let pending_rename_dropped = Arc::new(AtomicBool::new(false));
    let provider = Arc::new(RevokingProvider {
        memory: memory.clone(),
        valid: Arc::new(AtomicBool::new(true)),
        writes: Arc::new(AtomicUsize::new(0)),
        pending_rename_dropped: Some(pending_rename_dropped.clone()),
    });
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_cancellation_token(cancellation.token());

    let error = runmat_filesystem::with_provider_override(provider, || {
        let mut call = Box::pin(super::super::super::movefile::movefile_builtin(
            Some(runtime),
            vec![
                runmat_value::Value::from("source.bin"),
                runmat_value::Value::from("target.bin"),
            ],
        ));
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(matches!(call.as_mut().poll(&mut context), Poll::Pending));
        cancellation.cancel(runmat_execution::CancellationReason::User);
        match call.as_mut().poll(&mut context) {
            Poll::Ready(Err(error)) => error,
            other => panic!("cancelled move must finish with an error, got {other:?}"),
        }
    });

    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert!(pending_rename_dropped.load(Ordering::SeqCst));
    assert_eq!(
        memory.read_project_path("/selected/source.bin").unwrap(),
        b"source"
    );
    assert!(memory.read_project_path("/selected/target.bin").is_err());
}
