use runmat_accelerate_api::allocation::WeakAllocationLease;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
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
    ControlAlias,
    Siblings,
    ExplicitSibling,
    LowerShape,
    Pending,
    Unsupported,
    UnsupportedAfterAdmission,
    NonFinite,
}
#[derive(Clone, Copy)]
pub enum Upload {
    Good,
    Unsupported,
    Failure,
    Alias,
    SecondFailure,
    Logical,
    Foreign,
    Dead,
    Complex,
    Integer,
    Previous,
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
    pub stage: AtomicUsize,
    pub fail_stage: AtomicUsize,
    pub pending_stage: AtomicUsize,
    pub malformed_stage: AtomicUsize,
    pub fail_download: AtomicBool,
    pub pending_download: AtomicBool,
    pub raw_uploads: AtomicUsize,
    last_upload: Mutex<Option<(WeakAllocationLease, GpuTensorWireDescriptor)>>,
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
            stage: AtomicUsize::new(0),
            fail_stage: AtomicUsize::new(usize::MAX),
            pending_stage: AtomicUsize::new(usize::MAX),
            malformed_stage: AtomicUsize::new(usize::MAX),
            fail_download: AtomicBool::new(false),
            pending_download: AtomicBool::new(false),
            raw_uploads: AtomicUsize::new(0),
            last_upload: Mutex::new(None),
        }))
    }
    fn step(&self) -> anyhow::Result<usize> {
        let stage = self.stage.fetch_add(1, Ordering::SeqCst);
        if stage == self.fail_stage.load(Ordering::SeqCst) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "spectral continuation failure",
            )
            .into());
        }
        Ok(stage)
    }
    fn shaped(
        &self,
        input: &GpuTensorHandle,
        mut shape: Vec<usize>,
        storage: GpuTensorStorage,
        stage: usize,
    ) -> anyhow::Result<GpuTensorHandle> {
        if stage == self.malformed_stage.load(Ordering::SeqCst) {
            shape[0] += 1;
        }
        let owner = if matches!(self.mode, Mode::Foreign | Mode::Dead) {
            &self.foreign
        } else {
            &self.owner
        };
        let output = if matches!(self.mode, Mode::Logical) {
            owner.allocate_logical_zeroed(shape, ProviderPrecision::F64)?
        } else {
            owner.allocate_zeroed(
                shape,
                if matches!(self.mode, Mode::Class) {
                    NumericElementType::F32
                } else {
                    input.physical().element_type()
                },
                storage,
            )?
        };
        if matches!(self.mode, Mode::Dead) {
            output.probe.mark_unavailable();
        }
        self.produced.lock().unwrap().push(output.probe);
        Ok(output.handle)
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
        self.raw_uploads.fetch_add(1, Ordering::SeqCst);
        panic!("spectral outputs require native typed uploads")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("native download required")
    }
    fn upload_numeric_raw(
        &self,
        host: &HostNumericTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        let upload = self.uploads.fetch_add(1, Ordering::SeqCst);
        if matches!(self.upload, Upload::SecondFailure) && upload == 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "second upload failure",
            )
            .into());
        }
        let stage = self.step()?;
        match self.upload {
            Upload::Unsupported => {
                return Err(
                    unsupported_provider_operation("fixture typed upload unavailable").into(),
                );
            }
            Upload::Failure => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "typed upload failure",
                )
                .into());
            }
            Upload::Alias => return Ok(self.borrowed.lock().unwrap().as_ref().unwrap().clone()),
            Upload::Previous if upload != 0 => {
                let previous = self.last_upload.lock().unwrap();
                let (lease, wire) = previous.as_ref().expect("first coordinate");
                let handle = lease.acquire_tensor(wire)?;
                return Ok(handle.with_view(handle.view().clone().with_shape(host.shape.to_vec()))?);
            }
            Upload::Good
            | Upload::SecondFailure
            | Upload::Logical
            | Upload::Foreign
            | Upload::Dead
            | Upload::Complex
            | Upload::Integer
            | Upload::Previous => {}
        }
        if matches!(
            self.upload,
            Upload::Logical | Upload::Foreign | Upload::Dead | Upload::Complex | Upload::Integer
        ) {
            let owner = if matches!(self.upload, Upload::Foreign) {
                &self.foreign
            } else {
                &self.owner
            };
            let allocation = if matches!(self.upload, Upload::Logical) {
                owner.allocate_logical_zeroed(host.shape.to_vec(), ProviderPrecision::F64)?
            } else {
                owner.allocate_zeroed(
                    host.shape.to_vec(),
                    if matches!(self.upload, Upload::Integer) {
                        NumericElementType::I64
                    } else {
                        NumericElementType::F64
                    },
                    if matches!(self.upload, Upload::Complex) {
                        GpuTensorStorage::ComplexInterleaved
                    } else {
                        GpuTensorStorage::Real
                    },
                )?
            };
            if matches!(self.upload, Upload::Dead) {
                allocation.probe.mark_unavailable();
            }
            self.produced.lock().unwrap().push(allocation.probe);
            return Ok(allocation.handle);
        }
        let data = match host.data {
            HostNumericDataView::F32(data) => HostNumericDataOwned::F32(data.to_vec()),
            HostNumericDataView::F64(data) => HostNumericDataOwned::F64(data.to_vec()),
            _ => panic!("floating signal output"),
        };
        let output = self.owner.allocate_numeric(HostNumericTensorOwned {
            data,
            shape: if stage == self.malformed_stage.load(Ordering::SeqCst) {
                host.shape.iter().rev().copied().collect()
            } else {
                host.shape.to_vec()
            },
            storage: host.storage,
        })?;
        if matches!(self.upload, Upload::Previous) {
            *self.last_upload.lock().unwrap() = Some((
                output.handle.allocation_lease().downgrade(),
                output.handle.wire_descriptor(),
            ));
        }
        self.produced.lock().unwrap().push(output.probe);
        Ok(output.handle)
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            if self.fail_download.load(Ordering::SeqCst) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "signal download failed",
                )
                .into());
            }
            if self.pending_download.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            input.allocation_lease().validate_live()?;
            let probes = self
                .source
                .lock()
                .unwrap()
                .iter()
                .cloned()
                .chain(self.produced.lock().unwrap().iter().cloned())
                .collect::<Vec<_>>();
            for probe in probes {
                if let Ok(host) = probe.with_data(input, |data| {
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
                }) {
                    return Ok(host);
                }
            }
            Err(anyhow::anyhow!("unknown fixture allocation"))
        })
    }
    fn uniform_spectral_estimate<'a>(
        &'a self,
        request: &'a ProviderSpectralRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderSpectralResult> {
        Box::pin(async move {
            self.hooks.fetch_add(1, Ordering::SeqCst);
            let stage = self.step()?;
            if matches!(self.mode, Mode::Unsupported) {
                return Err(unsupported_provider_operation("spectral fixture unsupported").into());
            }
            let rows = match request.range {
                ProviderSpectralRange::Onesided => request.nfft / 2 + 1,
                _ => request.nfft,
            };
            let cols = request.frame_count;
            let s = self.shaped(
                request.input,
                vec![rows, cols],
                GpuTensorStorage::ComplexInterleaved,
                stage,
            )?;
            let ps = if matches!(self.mode, Mode::ControlAlias) {
                self.borrowed.lock().unwrap().as_ref().unwrap().clone()
            } else if matches!(self.mode, Mode::Siblings) {
                s.clone()
            } else if matches!(self.mode, Mode::Alias) {
                request.input.clone()
            } else {
                self.shaped(
                    request.input,
                    vec![rows, cols],
                    GpuTensorStorage::Real,
                    stage,
                )?
            };
            if stage == self.pending_stage.load(Ordering::SeqCst) {
                std::future::pending::<()>().await;
            }
            Ok(ProviderSpectralResult {
                s,
                ps,
                rows: if matches!(self.mode, Mode::Shape) {
                    rows + 1
                } else {
                    rows
                },
                cols,
            })
        })
    }
    fn reshape(&self, input: &GpuTensorHandle, shape: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        let stage = self.step()?;
        if stage == self.malformed_stage.load(Ordering::SeqCst) {
            return self.shaped(input, shape.to_vec(), GpuTensorStorage::Real, stage);
        }
        Ok(input.with_view(input.view().clone().with_shape(shape.to_vec()))?)
    }
    fn reduce_mean_nd<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        dims: &'a [usize],
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            let result: anyhow::Result<GpuTensorHandle> = async move {
                let stage = self.step()?;
                let mut shape = input.shape().to_vec();
                for &dim in dims {
                    shape[dim] = 1;
                }
                let output = self.shaped(input, shape, GpuTensorStorage::Real, stage)?;
                if stage == self.pending_stage.load(Ordering::SeqCst) {
                    std::future::pending::<()>().await;
                }
                Ok(output)
            }
            .await;
            result.map(Some).map_err(|source| {
                runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
            })
        })
    }
    fn supports_signal_hilbert(&self, _: &ProviderHilbertRequest<'_>) -> bool {
        // This fixture deliberately exercises malformed and terminal hook
        // outputs. Its unavailable mode instead exercises preflight fallback.
        !matches!(self.mode, Mode::Unsupported)
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
    fn supports_signal_envelope(&self, _: &ProviderEnvelopeRequest<'_>) -> bool {
        !matches!(self.mode, Mode::Unsupported)
    }
    fn signal_envelope<'a>(
        &'a self,
        request: &'a ProviderEnvelopeRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderEnvelopeResult> {
        Box::pin(async move {
            self.hooks.fetch_add(1, Ordering::SeqCst);
            if matches!(
                self.mode,
                Mode::Unsupported | Mode::UnsupportedAfterAdmission
            ) {
                return Err(unsupported_provider_operation("fixture envelope unavailable").into());
            }
            if matches!(self.mode, Mode::NonFinite) {
                return Err(ProviderExecutionError::from_source(
                    EnvelopeSignalDomainError::NonFinite,
                )
                .after_effect());
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
            let lower = if matches!(self.mode, Mode::ExplicitSibling) {
                lower
                    .with_provenance(GpuHandleProvenance::Explicit)
                    .map_err(ProviderExecutionError::from_source)?
            } else if matches!(self.mode, Mode::LowerShape) {
                lower
                    .with_view(lower.view().clone().with_shape(vec![1, 4]))
                    .map_err(ProviderExecutionError::from_source)?
            } else {
                lower
            };
            Ok(ProviderEnvelopeResult { upper, lower })
        })
    }
}
