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

mod cases;
