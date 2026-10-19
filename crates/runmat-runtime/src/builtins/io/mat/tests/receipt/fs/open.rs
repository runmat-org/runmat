use super::*;
use futures::channel::oneshot;
use runmat_filesystem::{FsResourceContract, FsResourceLease, FsResourceRevision};
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

struct TestFsLease(AtomicBool);
impl FsResourceLease for TestFsLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("mat-gated-test", b"v1"),
            opened_input_handle: None,
            opened_output_handle: None,
        }
    }

    fn is_valid(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

pub(in crate::builtins::io::mat) struct GatedOpenFs {
    inner: runmat_filesystem::MemoryFsProvider,
    cwd: PathBuf,
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    opens: AtomicUsize,
    reads: Arc<AtomicUsize>,
    revoke_on_read: Mutex<Option<Arc<AtomicBool>>>,
    fs_lease: Arc<TestFsLease>,
}

impl GatedOpenFs {
    pub(in crate::builtins::io::mat) fn new(
        cwd: &str,
        filename: &str,
        bytes: &[u8],
    ) -> (Self, oneshot::Sender<()>) {
        let inner = runmat_filesystem::MemoryFsProvider::with_current_dir(cwd);
        inner
            .write_project_path(filename, bytes)
            .expect("seed gated MAT source");
        let (sender, receiver) = oneshot::channel();
        (
            Self {
                inner,
                cwd: PathBuf::from(cwd),
                gate: Mutex::new(Some(receiver)),
                opens: AtomicUsize::new(0),
                reads: Arc::new(AtomicUsize::new(0)),
                revoke_on_read: Mutex::new(None),
                fs_lease: Arc::new(TestFsLease(AtomicBool::new(true))),
            },
            sender,
        )
    }

    pub(in crate::builtins::io::mat) fn opens(&self) -> usize {
        self.opens.load(Ordering::SeqCst)
    }

    pub(in crate::builtins::io::mat) fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }

    pub(in crate::builtins::io::mat) fn revoke_on_read(&self, live: Arc<AtomicBool>) {
        *self
            .revoke_on_read
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(live);
    }

    pub(in crate::builtins::io::mat) fn revoke_fs(&self) {
        self.fs_lease.0.store(false, Ordering::SeqCst);
    }
}

struct CountedHandle {
    inner: Box<dyn FileHandle>,
    reads: Arc<AtomicUsize>,
    revoke_on_read: Option<Arc<AtomicBool>>,
}

impl Read for CountedHandle {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        let read = self.inner.read(buffer)?;
        if read != 0 {
            if let Some(live) = self.revoke_on_read.as_ref() {
                live.store(false, Ordering::SeqCst);
            }
        }
        Ok(read)
    }
}
impl Write for CountedHandle {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.inner.write(buffer)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
impl Seek for CountedHandle {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.inner.seek(position)
    }
}
#[async_trait(?Send)]
impl FileHandle for CountedHandle {
    async fn metadata_async(&self) -> io::Result<FsMetadata> {
        self.inner.metadata_async().await
    }
}

#[async_trait(?Send)]
impl FsProvider for GatedOpenFs {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.inner.path_namespace()
    }

    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(self.fs_lease.clone())
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(self.cwd.clone())
    }
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Ok(Box::new(CountedHandle {
            inner: self.inner.open(path, flags)?,
            reads: self.reads.clone(),
            revoke_on_read: self
                .revoke_on_read
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone(),
        }))
    }
    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        let gate = self
            .gate
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(gate) = gate {
            gate.await
                .map_err(|_| io::Error::other("MAT open gate dropped"))?;
        }
        self.open(path, flags)
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.inner.read(path).await
    }
    async fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.inner.write(path, bytes).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_file(path).await
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
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.inner.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.inner.rename(from, to).await
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.inner.set_readonly(path, readonly).await
    }
}
