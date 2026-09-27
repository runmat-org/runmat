use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Good,
    Shape,
    Class,
    Logical,
    Foreign,
    Dead,
    Alias,
    Siblings,
    Pending,
    Unsupported,
}
#[derive(Clone, Copy)]
pub enum Upload {
    Good,
    Unsupported,
    Failure,
    Alias,
}

pub struct Provider {
    pub owner: TestAllocationOwner,
    foreign: TestAllocationOwner,
    pub mode: Mode,
    pub upload: Upload,
    pub hooks: AtomicUsize,
    pub downloads: AtomicUsize,
    pub uploads: AtomicUsize,
    pub produced: Mutex<Vec<RetirementProbe>>,
    pub source: Mutex<Option<RetirementProbe>>,
    pub borrowed: Mutex<Option<GpuTensorHandle>>,
}
impl Provider {
    pub fn new(mode: Mode, upload: Upload) -> &'static Self {
        Box::leak(Box::new(Self {
            owner: TestAllocationOwner::new(),
            foreign: TestAllocationOwner::new(),
            mode,
            upload,
            hooks: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
            uploads: AtomicUsize::new(0),
            produced: Mutex::new(Vec::new()),
            source: Mutex::new(None),
            borrowed: Mutex::new(None),
        }))
    }
    fn output(
        &self,
        input: &GpuTensorHandle,
        storage: GpuTensorStorage,
    ) -> anyhow::Result<GpuTensorHandle> {
        if matches!(self.mode, Mode::Alias) {
            return Ok(input.clone());
        }
        let shape = if matches!(self.mode, Mode::Shape) {
            vec![1, input.shape().iter().product()]
        } else {
            input.shape().to_vec()
        };
        let element = if matches!(self.mode, Mode::Class) {
            NumericElementType::F32
        } else {
            input.physical().element_type()
        };
        let owner = if matches!(self.mode, Mode::Foreign | Mode::Dead) {
            &self.foreign
        } else {
            &self.owner
        };
        let allocation = if matches!(self.mode, Mode::Logical) {
            owner.allocate_logical_zeroed(shape, ProviderPrecision::F64)?
        } else {
            owner.allocate_zeroed(shape, element, storage)?
        };
        if matches!(self.mode, Mode::Dead) {
            allocation.probe.mark_unavailable();
        }
        self.produced.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    pub fn assert_retired(&self) {
        for probe in self.produced.lock().unwrap().iter() {
            assert_eq!(probe.retirements(), 1);
        }
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "signal owning boundary fixture".into()
    }
    fn precision(&self) -> ProviderPrecision {
        ProviderPrecision::F64
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("native upload required")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("native download required")
    }
    fn upload_numeric_raw(&self, host: &HostNumericTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.uploads.fetch_add(1, Ordering::SeqCst);
        match self.upload {
            Upload::Unsupported => {
                return Err(unsupported_provider_operation(
                    "fixture typed upload unavailable",
                ))
            }
            Upload::Failure => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "typed upload failure",
                )
                .into())
            }
            Upload::Alias => return Ok(self.borrowed.lock().unwrap().as_ref().unwrap().clone()),
            Upload::Good => {}
        }
        let data = match host.data {
            HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
            HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
            _ => panic!("floating signal output"),
        };
        let output = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: host.shape.to_vec(),
            storage: host.storage,
        })?;
        self.produced.lock().unwrap().push(output.probe);
        Ok(output.handle)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            self.source
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .with_data(input, |data| {
                    let data = match data {
                        HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
                        HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
                        _ => panic!("floating signal input"),
                    };
                    HostNumericTensorOwned {
                        data,
                        shape: input.shape().to_vec(),
                        storage: input.physical().storage(),
                    }
                })
        })
    }
    fn signal_hilbert<'a>(
        &'a self,
        request: &'a ProviderHilbertRequest<'a>,
    ) -> ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.hooks.fetch_add(1, Ordering::SeqCst);
            if matches!(self.mode, Mode::Unsupported) {
                return Err(unsupported_provider_operation("fixture Hilbert unavailable").into());
            }
            let output = self.output(request.input, GpuTensorStorage::ComplexInterleaved)?;
            if matches!(self.mode, Mode::Pending) {
                std::future::pending::<()>().await;
            }
            Ok(output)
        })
    }
    fn signal_envelope<'a>(
        &'a self,
        request: &'a ProviderEnvelopeRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderEnvelopeResult> {
        Box::pin(async move {
            self.hooks.fetch_add(1, Ordering::SeqCst);
            if matches!(self.mode, Mode::Unsupported) {
                return Err(unsupported_provider_operation("fixture envelope unavailable").into());
            }
            let upper = self.output(request.input, GpuTensorStorage::Real)?;
            let lower = if matches!(self.mode, Mode::Siblings) {
                upper.clone()
            } else {
                self.output(request.input, GpuTensorStorage::Real)?
            };
            if matches!(self.mode, Mode::Pending) {
                std::future::pending::<()>().await;
            }
            Ok(ProviderEnvelopeResult { upper, lower })
        })
    }
}
