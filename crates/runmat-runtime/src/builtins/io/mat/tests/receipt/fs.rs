use async_trait::async_trait;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

mod open;
pub(in crate::builtins::io::mat) use open::GatedOpenFs;

pub(in crate::builtins::io::mat) struct PendingWriteFs {
    inner: runmat_filesystem::MemoryFsProvider,
    writes: AtomicUsize,
    abandoned: AtomicUsize,
    cwd: PathBuf,
}

impl PendingWriteFs {
    pub(in crate::builtins::io::mat) fn new(cwd: &str) -> Self {
        Self {
            inner: runmat_filesystem::MemoryFsProvider::with_current_dir(cwd),
            writes: AtomicUsize::new(0),
            abandoned: AtomicUsize::new(0),
            cwd: PathBuf::from(cwd),
        }
    }

    pub(in crate::builtins::io::mat) fn writes(&self) -> usize {
        self.writes.load(Ordering::SeqCst)
    }

    pub(in crate::builtins::io::mat) fn abandoned(&self) -> usize {
        self.abandoned.load(Ordering::SeqCst)
    }
}

struct PendingWrite<'a>(&'a AtomicUsize);
impl Drop for PendingWrite<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn unused() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "unused MAT receipt fixture operation",
    )
}

#[async_trait(?Send)]
impl FsProvider for PendingWriteFs {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.inner.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(self.cwd.clone())
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.inner.open(path, flags)
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.inner.read(path).await
    }

    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        let _pending = PendingWrite(&self.abandoned);
        std::future::pending().await
    }

    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unused())
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
