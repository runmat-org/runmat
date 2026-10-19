use super::*;

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Normal,
    Unsupported,
    Failure,
    Shape,
    Class,
    Complex,
    Logical,
    Foreign,
    UploadUnsupported,
    UploadFailure,
    UploadAlias,
    ConjAlias,
    ConjOriginal,
    ConjUnavailable,
    ConjFailure,
    Identity,
    Pending,
}

pub(super) struct Provider {
    pub inner: InProcessProvider,
    foreign: InProcessProvider,
    mode: Mode,
    transpose: AtomicUsize,
    conj: AtomicUsize,
    downloads: AtomicUsize,
    uploads: AtomicUsize,
    pub original: Mutex<Option<allocation::WeakGpuTensorHandle>>,
    pub produced: Mutex<Vec<allocation::WeakAllocationLease>>,
}
impl Provider {
    pub fn new(mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            inner: InProcessProvider::new(),
            foreign: InProcessProvider::new(),
            mode,
            transpose: AtomicUsize::new(0),
            conj: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            original: Mutex::new(None),
            produced: Mutex::new(vec![]),
        }))
    }
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        (
            self.transpose.load(Ordering::SeqCst),
            self.conj.load(Ordering::SeqCst),
            self.downloads.load(Ordering::SeqCst),
            self.uploads.load(Ordering::SeqCst),
        )
    }
    fn retain_probe(&self, output: &GpuTensorHandle) {
        self.produced
            .lock()
            .unwrap()
            .push(output.allocation_lease().downgrade());
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        "transpose contract owner".into()
    }
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        match self.mode {
            Mode::UploadUnsupported => {
                Err(unsupported_provider_operation("transpose restoration unsupported").into())
            }
            Mode::UploadFailure => {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "restore failure").into())
            }
            Mode::UploadAlias => Ok(self
                .original
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .acquire()?
                .with_view(GpuTensorView::contiguous(host.shape.to_vec()))?),
            _ => self.inner.upload_numeric_raw(host),
        }
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::SeqCst);
        self.inner.download_numeric(input)
    }
    fn transpose(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        self.transpose.fetch_add(1, Ordering::SeqCst);
        let output = match self.mode {
            Mode::Identity => input.clone(),
            Mode::Unsupported
            | Mode::UploadUnsupported
            | Mode::UploadFailure
            | Mode::UploadAlias => {
                return Err(unsupported_provider_operation("transpose unsupported"))
            }
            Mode::Failure => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "transpose not supported by provider",
                )
                .into())
            }
            Mode::Shape => self
                .inner
                .transpose(input)?
                .with_view(GpuTensorView::contiguous(vec![6, 1]))?,
            Mode::Class => self.inner.upload_numeric_host(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.; 6]),
                shape: &[3, 2],
                storage: GpuTensorStorage::Real,
            })?,
            Mode::Complex => self.inner.upload_numeric_host(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[0.; 12]),
                shape: &[3, 2],
                storage: GpuTensorStorage::ComplexInterleaved,
            })?,
            Mode::Logical => self
                .inner
                .upload_logical_raw(&HostLogicalTensorView {
                    data: &[0; 6],
                    shape: &[3, 2],
                })
                .expect("fixed logical fixture upload must succeed"),
            Mode::Foreign => self
                .foreign
                .transpose(&self.foreign.upload(&HostTensorView {
                    data: &[0.; 6],
                    shape: &[2, 3],
                })?)?,
            _ => self.inner.transpose(input)?,
        };
        self.retain_probe(&output);
        Ok(output)
    }
    fn unary_conj<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        self.conj.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            match self.mode {
                Mode::Pending => std::future::pending().await,
                Mode::ConjUnavailable => Ok(None),
                Mode::ConjFailure => Err(ProviderExecutionError::from(
                    unsupported_provider_operation("conjugation operational cause"),
                )
                .with_unproven_effects()),
                Mode::ConjAlias => Ok(Some(input.clone())),
                Mode::ConjOriginal => Ok(Some(
                    self.original
                        .lock()
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .acquire()?
                        .with_view(GpuTensorView::contiguous(input.shape().to_vec()))?,
                )),
                _ => self.inner.unary_conj(input).await,
            }
        })
    }
}
