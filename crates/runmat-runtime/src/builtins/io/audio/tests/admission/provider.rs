use super::*;
use async_trait::async_trait;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(super) struct CountedFs {
    pub(super) inner: runmat_filesystem::MemoryFsProvider,
    pub(super) opens: AtomicUsize,
    pub(super) reads: AtomicUsize,
    pub(super) metadata: AtomicUsize,
    pub(super) pending_open: AtomicBool,
    cwd: PathBuf,
}

impl CountedFs {
    pub(super) fn new(cwd: &str) -> Self {
        Self {
            inner: runmat_filesystem::MemoryFsProvider::with_current_dir(cwd),
            opens: AtomicUsize::new(0),
            reads: AtomicUsize::new(0),
            metadata: AtomicUsize::new(0),
            pending_open: AtomicBool::new(false),
            cwd: PathBuf::from(cwd),
        }
    }

    pub(super) fn effects(&self) -> (usize, usize, usize) {
        (
            self.opens.load(Ordering::SeqCst),
            self.reads.load(Ordering::SeqCst),
            self.metadata.load(Ordering::SeqCst),
        )
    }
}

fn unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "unused counted audio fixture route",
    )
}

#[async_trait(?Send)]
impl FsProvider for CountedFs {
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(self.cwd.clone())
    }
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        self.inner.open(path, flags)
    }
    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        if self.pending_open.load(Ordering::SeqCst) {
            self.opens.fetch_add(1, Ordering::SeqCst);
            std::future::pending().await
        } else {
            self.open(path, flags)
        }
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.read(path).await
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.metadata.fetch_add(1, Ordering::SeqCst);
        self.inner.metadata(path).await
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unsupported())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unsupported())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unsupported())
    }
}
