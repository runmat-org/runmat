use futures::channel::oneshot;
use runmat_filesystem as vfs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

pub(super) struct GatedRemoveProvider {
    inner: vfs::MemoryFsProvider,
    entered: Mutex<Option<oneshot::Sender<()>>>,
    release: Mutex<Option<oneshot::Receiver<()>>>,
    closed: Arc<AtomicUsize>,
}

impl GatedRemoveProvider {
    pub(super) fn new() -> (
        Arc<Self>,
        oneshot::Receiver<()>,
        oneshot::Sender<()>,
        Arc<AtomicUsize>,
    ) {
        let (entered, observed) = oneshot::channel();
        let (release, pending) = oneshot::channel();
        let closed = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(Self {
                inner: vfs::MemoryFsProvider::new(),
                entered: Mutex::new(Some(entered)),
                release: Mutex::new(Some(pending)),
                closed: closed.clone(),
            }),
            observed,
            release,
            closed,
        )
    }

    pub(super) fn read_project_path(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.inner.read_project_path(path)
    }
}

struct ProbedHandle {
    inner: Box<dyn vfs::FileHandle>,
    closed: Arc<AtomicUsize>,
}

impl Read for ProbedHandle {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.inner.read(bytes)
    }
}

impl Write for ProbedHandle {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.inner.write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl Seek for ProbedHandle {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}

#[async_trait::async_trait(?Send)]
impl vfs::FileHandle for ProbedHandle {}

impl Drop for ProbedHandle {
    fn drop(&mut self) {
        self.closed.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait::async_trait(?Send)]
impl vfs::FsProvider for GatedRemoveProvider {
    fn open(&self, path: &Path, flags: &vfs::OpenFlags) -> io::Result<Box<dyn vfs::FileHandle>> {
        Ok(Box::new(ProbedHandle {
            inner: vfs::FsProvider::open(&self.inner, path, flags)?,
            closed: self.closed.clone(),
        }))
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        vfs::FsProvider::read(&self.inner, path).await
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        vfs::FsProvider::write(&self.inner, path, data).await
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        let release = self.release.lock().unwrap().take();
        if let Some(release) = release {
            if let Some(entered) = self.entered.lock().unwrap().take() {
                let _ = entered.send(());
            }
            release
                .await
                .map_err(|_| io::Error::other("rollback gate abandoned"))?;
        }
        vfs::FsProvider::remove_file(&self.inner, path).await
    }

    async fn metadata(&self, path: &Path) -> io::Result<vfs::FsMetadata> {
        vfs::FsProvider::metadata(&self.inner, path).await
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<vfs::FsMetadata> {
        vfs::FsProvider::symlink_metadata(&self.inner, path).await
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<vfs::DirEntry>> {
        vfs::FsProvider::read_dir(&self.inner, path).await
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        vfs::FsProvider::canonicalize(&self.inner, path).await
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        vfs::FsProvider::create_dir(&self.inner, path).await
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        vfs::FsProvider::create_dir_all(&self.inner, path).await
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        vfs::FsProvider::remove_dir(&self.inner, path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        vfs::FsProvider::remove_dir_all(&self.inner, path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        vfs::FsProvider::rename(&self.inner, from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        vfs::FsProvider::set_readonly(&self.inner, path, readonly).await
    }
}
