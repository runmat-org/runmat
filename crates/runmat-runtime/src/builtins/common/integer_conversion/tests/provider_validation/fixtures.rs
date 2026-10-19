use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, GpuTensorHandle, HostIntegerTensorView, HostTensorView,
    IntegerElementType, ProviderPrecision,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
pub(super) enum ResultMode {
    WrongClass,
    ForeignOwner,
    Alias,
    Failure,
    Unsupported,
    Pending,
}

pub(super) struct Provider {
    pub(super) inner: InProcessProvider,
    mode: Mutex<ResultMode>,
    pub(super) calls: AtomicUsize,
    pub(super) downloads: AtomicUsize,
    pub(super) created: Mutex<Vec<runmat_accelerate_api::allocation::WeakAllocationLease>>,
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
    fn upload(&self, view: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(view)
    }
    fn upload_integer(&self, view: &HostIntegerTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload_integer(view)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download(handle)
    }
    fn upload_numeric_raw(
        &self,
        view: &runmat_accelerate_api::HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        self.inner.upload_numeric_raw(view)
    }
    fn download_numeric<'a>(
        &'a self,
        handle: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelNumericDownloadFuture<'a> {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.inner.download_numeric(handle)
    }
    fn cast_to_integer<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        target: IntegerElementType,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let mode = *self.mode.lock().unwrap();
        Box::pin(async move {
            match mode {
                ResultMode::Pending => {
                    runmat_accelerate_api::clear_residency(input)
                        .expect("live fixture residency can be cleared");
                    futures::future::pending().await
                }
                ResultMode::Alias => Ok(input.clone()),
                ResultMode::Failure => Err(runmat_accelerate_api::ProviderExecutionError::from(
                    anyhow::anyhow!("injected cast execution failure"),
                )),
                ResultMode::Unsupported => Err(
                    runmat_accelerate_api::unsupported_provider_operation("cast unavailable")
                        .into(),
                ),
                ResultMode::WrongClass | ResultMode::ForeignOwner => {
                    let output = if matches!(mode, ResultMode::WrongClass) {
                        self.inner.upload(&HostTensorView {
                            data: &[3.0, -1.0],
                            shape: input.shape(),
                        })?
                    } else {
                        let foreign = InProcessProvider::new();
                        let foreign_input = foreign.upload(&HostTensorView {
                            data: &[3.0, -1.0],
                            shape: input.shape(),
                        })?;
                        foreign.cast_to_integer(&foreign_input, target).await?
                    };
                    self.created
                        .lock()
                        .unwrap()
                        .push(output.allocation_lease().downgrade());
                    Ok(output)
                }
            }
        })
    }
}

pub(super) fn provider(mode: ResultMode) -> &'static Provider {
    Box::leak(Box::new(Provider {
        inner: InProcessProvider::new(),
        mode: Mutex::new(mode),
        calls: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        created: Mutex::new(Vec::new()),
    }))
}

pub(super) fn input(provider: &Provider) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data: &[3.5, -1.0],
            shape: &[1, 2],
        })
        .unwrap()
}
