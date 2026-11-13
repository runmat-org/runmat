use runmat_accelerate_api::*;
use std::sync::Mutex;

#[derive(Clone, Copy)]
pub enum Mode {
    Supported,
    Unsupported,
    Failure,
    Alias,
}
pub struct Provider {
    pub inner: &'static dyn AccelProvider,
    pub mode: Mode,
    pub alias: Mutex<Option<GpuTensorHandle>>,
    pub trace: Mutex<Vec<&'static str>>,
}
impl Provider {
    pub fn new(inner: &'static dyn AccelProvider, mode: Mode) -> &'static Self {
        Box::leak(Box::new(Self {
            inner,
            mode,
            alias: Mutex::new(None),
            trace: Mutex::new(vec![]),
        }))
    }
    fn intercept(&self, label: &'static str) -> anyhow::Result<Option<GpuTensorHandle>> {
        self.trace.lock().unwrap().push(label);
        match self.mode {
            Mode::Supported => Ok(None),
            Mode::Unsupported => Err(unsupported_provider_operation(label)),
            Mode::Failure => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "restoration transport failure",
            )
            .into()),
            Mode::Alias => Ok(Some(self.alias.lock().unwrap().as_ref().unwrap().clone())),
        }
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.inner.device_id()
    }
    fn device_info(&self) -> String {
        self.inner.device_info()
    }
    fn precision(&self) -> ProviderPrecision {
        self.inner.precision()
    }
    fn upload(&self, host: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        match self.intercept("numeric-upload")? {
            Some(alias) => Ok(alias),
            None => self.inner.upload_numeric_raw(host),
        }
    }
    fn upload_logical_raw(
        &self,
        host: &HostLogicalTensorView<'_>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        match self.intercept("logical-upload") {
            Ok(Some(alias)) => Ok(alias),
            Ok(None) => self.inner.upload_logical_raw(host),
            Err(error) if matches!(self.mode, Mode::Unsupported) => Err(error.into()),
            Err(error) => {
                Err(runmat_accelerate_api::ProviderExecutionError::from(error)
                    .with_unproven_effects())
            }
        }
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.trace.lock().unwrap().push("download");
        self.inner.download(input)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        self.trace.lock().unwrap().push("download");
        self.inner.download_numeric(input)
    }
}

pub fn assert_upload_failure(error: &runmat_runtime::RuntimeError, alias: bool) {
    assert_eq!(
        error.gpu_gather_retry,
        runmat_runtime::GpuGatherRetry::Never
    );
    let upload = error
        .source
        .as_ref()
        .unwrap()
        .downcast_ref::<NumericUploadError>()
        .expect("typed retained upload failure");
    if alias {
        assert!(matches!(upload, NumericUploadError::Invalid(_)));
    } else {
        let NumericUploadError::Provider(cause) = upload else {
            panic!("provider cause")
        };
        assert_eq!(
            std::error::Error::source(cause)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .unwrap()
                .kind(),
            std::io::ErrorKind::BrokenPipe
        );
    }
}
