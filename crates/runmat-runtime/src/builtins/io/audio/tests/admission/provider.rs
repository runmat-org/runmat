use async_trait::async_trait;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
use std::io::{self, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub(super) struct CountedFs {
    pub(super) inner: runmat_filesystem::MemoryFsProvider,
    pub(super) opens: AtomicUsize,
    pub(super) reads: AtomicUsize,
    pub(super) metadata: AtomicUsize,
    pub(super) pending_open: AtomicBool,
    pub(super) pending_read: AtomicBool,
    pub(super) pending_metadata: Arc<AtomicBool>,
    pub(super) pending_provider_metadata: AtomicBool,
    pub(super) unsupported_open: AtomicBool,
    pub(super) cancel_on_read: Arc<Mutex<Option<CancellationSource>>>,
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
            pending_read: AtomicBool::new(false),
            pending_metadata: Arc::new(AtomicBool::new(false)),
            pending_provider_metadata: AtomicBool::new(false),
            unsupported_open: AtomicBool::new(false),
            cancel_on_read: Arc::new(Mutex::new(None)),
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

struct MetadataHandle {
    inner: Box<dyn FileHandle>,
    pending: Arc<AtomicBool>,
    cancel_on_read: Arc<Mutex<Option<CancellationSource>>>,
}

impl Read for MetadataHandle {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if let Some(source) = self
            .cancel_on_read
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            source.cancel(CancellationReason::User);
        }
        self.inner.read(buffer)
    }
}

impl Write for MetadataHandle {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.inner.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for MetadataHandle {
    fn seek(&mut self, position: io::SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}

#[async_trait(?Send)]
impl FileHandle for MetadataHandle {
    async fn metadata_async(&self) -> io::Result<FsMetadata> {
        if self.pending.load(Ordering::SeqCst) {
            std::future::pending().await
        } else {
            self.inner.metadata_async().await
        }
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
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.inner.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(self.cwd.clone())
    }
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        if self.unsupported_open.load(Ordering::SeqCst) {
            return Err(unsupported());
        }
        Ok(Box::new(MetadataHandle {
            inner: self.inner.open(path, flags)?,
            pending: self.pending_metadata.clone(),
            cancel_on_read: self.cancel_on_read.clone(),
        }))
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
        if self.pending_read.load(Ordering::SeqCst) {
            return std::future::pending().await;
        }
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
        if self.pending_provider_metadata.load(Ordering::SeqCst) {
            return std::future::pending().await;
        }
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
