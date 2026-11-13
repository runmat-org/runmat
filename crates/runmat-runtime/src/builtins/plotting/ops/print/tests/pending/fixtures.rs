use super::*;

pub(super) struct PausedProvider {
    pub(super) gate: Mutex<Option<oneshot::Receiver<()>>>,
    pub(super) started: Mutex<Option<oneshot::Sender<()>>>,
    pub(super) published: Arc<Mutex<Option<Vec<u8>>>>,
    pub(super) drops: Arc<AtomicUsize>,
    pub(super) lease: Arc<RevocableLease>,
}

pub(super) struct PausedDownloadProvider {
    pub(super) inner: runmat_accelerate::simple_provider::InProcessProvider,
    pub(super) gate: Mutex<Option<oneshot::Receiver<()>>>,
    pub(super) started: Mutex<Option<oneshot::Sender<()>>>,
    pub(super) downloads: AtomicUsize,
}

impl PausedDownloadProvider {
    async fn wait_for_download(&self) -> anyhow::Result<()> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        let gate = self
            .gate
            .lock()
            .unwrap()
            .take()
            .expect("one resident gather");
        self.started
            .lock()
            .unwrap()
            .take()
            .expect("gather start sender")
            .send(())
            .expect("public test awaits gather start");
        gate.await
            .map_err(|_| anyhow::anyhow!("test download abandoned"))
    }
}

impl AccelProvider for PausedDownloadProvider {
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }

    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.inner.download(handle)
    }

    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(host)
    }

    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.wait_for_download().await?;
            self.inner.download_numeric(handle).await
        })
    }

    fn download_integer<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> AccelIntegerDownloadFuture<'a> {
        Box::pin(async move {
            self.wait_for_download().await?;
            self.inner.download_integer(handle).await
        })
    }

    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }

    fn device_info(&self) -> String {
        self.inner.device_info()
    }
}

pub(super) struct RevocableLease(AtomicBool);

impl RevocableLease {
    pub(super) fn live() -> Arc<Self> {
        Arc::new(Self(AtomicBool::new(true)))
    }
    pub(super) fn revoke(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl FsResourceLease for RevocableLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("print.pending.fixture", b"v1"),
            opened_input_handle: None,
            opened_output_handle: None,
        }
    }
    fn is_valid(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

struct PausedStage {
    gate: Option<oneshot::Receiver<()>>,
    bytes: Vec<u8>,
    published: Arc<Mutex<Option<Vec<u8>>>>,
    drops: Arc<AtomicUsize>,
}

impl Drop for PausedStage {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait(?Send)]
impl FsStagedOutput for PausedStage {
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if let Some(gate) = self.gate.take() {
            gate.await
                .map_err(|_| io::Error::other("test stage abandoned"))?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    async fn flush_and_sync(&mut self) -> io::Result<()> {
        Ok(())
    }
    async fn publish(mut self: Box<Self>) -> Result<(), FsPublicationFailure> {
        *self.published.lock().unwrap() = Some(std::mem::take(&mut self.bytes));
        Ok(())
    }
}

fn unused() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "unused fixture operation")
}

#[async_trait(?Send)]
impl FsProvider for PausedProvider {
    fn path_namespace(&self) -> fs::FsPathNamespace {
        fs::FsPathNamespace::VirtualRooted
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }
    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(self.lease.clone())
    }
    fn begin_staged_replace(&self, _path: &Path) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        let gate = self
            .gate
            .lock()
            .unwrap()
            .take()
            .expect("one public print stage");
        self.started
            .lock()
            .unwrap()
            .take()
            .expect("stage start sender")
            .send(())
            .expect("public test awaits stage start");
        Ok(Some(Box::new(PausedStage {
            gate: Some(gate),
            bytes: Vec::new(),
            published: self.published.clone(),
            drops: self.drops.clone(),
        })))
    }
    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(unused())
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        Err(unused())
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unused())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unused())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unused())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unused())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unused())
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

pub(super) fn await_operation_start<F: Future>(started: oneshot::Receiver<()>, call: Pin<&mut F>) {
    match futures::executor::block_on(select(started, call)) {
        Either::Left((Ok(()), _pending_call)) => {}
        Either::Left((Err(_), _pending_call)) => panic!("selected operation start was abandoned"),
        Either::Right((_finished, _start)) => {
            panic!("public print finished before pending operation")
        }
    }
}
