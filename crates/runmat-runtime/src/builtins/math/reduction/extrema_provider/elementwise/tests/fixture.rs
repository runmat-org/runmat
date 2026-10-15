use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocation, TestAllocationOwner},
    AccelDownloadFuture, AccelNumericDownloadFuture, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorOwned, HostTensorView,
};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Normal,
    WrongValues,
    WrongFill,
    AliasFill,
    WrongRepmat,
    AliasValues,
    PendingValues,
    PendingDownload,
    FailedValues,
    UnsupportedPair,
    UnsupportedFill,
}

pub(super) struct Provider {
    owner: TestAllocationOwner,
    pub(super) records: Mutex<Vec<(u64, RetirementProbe)>>,
    mode: Mode,
    pub(super) values: AtomicUsize,
    pub(super) downloads: AtomicUsize,
    pub(super) fills: AtomicUsize,
    pub(super) repeats: AtomicUsize,
}
impl Provider {
    pub(super) fn allocate(&self, data: Vec<f64>, shape: Vec<usize>) -> TestAllocation {
        let allocation = self
            .owner
            .allocate_numeric(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(data),
                shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        self.records
            .lock()
            .unwrap()
            .push((allocation.handle.buffer_id(), allocation.probe.clone()));
        allocation
    }
    pub(super) fn read(&self, input: &GpuTensorHandle) -> anyhow::Result<Vec<f64>> {
        let probe = self
            .records
            .lock()
            .unwrap()
            .iter()
            .find(|(id, _)| *id == input.buffer_id())
            .unwrap()
            .1
            .clone();
        probe.with_data(input, |data| match data {
            HostNumericDataView::F64(data) => data.to_vec(),
            _ => panic!("F64 fixture"),
        })
    }
    fn output(
        &self,
        left: &GpuTensorHandle,
        right: &[f64],
        maximum: bool,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.values.fetch_add(1, Ordering::SeqCst);
        if matches!(self.mode, Mode::FailedValues) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "extrema value failed",
            )
            .into());
        }
        if matches!(self.mode, Mode::WrongValues) {
            return Ok(self.allocate(vec![0.0], vec![1, 1]).handle);
        }
        if matches!(self.mode, Mode::AliasValues) {
            return Ok(left.clone());
        }
        let left_data = self.read(left)?;
        assert_eq!(left_data.len(), right.len());
        let data = left_data
            .into_iter()
            .zip(right)
            .map(|(a, &b)| {
                if a.is_nan() {
                    a
                } else if b.is_nan() {
                    b
                } else if a == 0.0 && b == 0.0 {
                    if maximum {
                        if a.is_sign_positive() || b.is_sign_positive() {
                            0.0
                        } else {
                            -0.0
                        }
                    } else if a.is_sign_negative() || b.is_sign_negative() {
                        -0.0
                    } else {
                        0.0
                    }
                } else if (maximum && b > a) || (!maximum && b < a) {
                    b
                } else {
                    a
                }
            })
            .collect();
        Ok(self.allocate(data, left.shape().to_vec()).handle)
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "extrema stages and demand witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("no implicit upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("typed download only") })
    }
    fn download_numeric<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelNumericDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            if matches!(self.mode, Mode::PendingDownload) {
                std::future::pending::<()>().await;
            }
            Ok(HostNumericTensorOwned {
                data: HostNumericDataOwned::F64(self.read(input)?),
                shape: input.shape().to_vec(),
                storage: GpuTensorStorage::Real,
            })
        })
    }
    fn elem_min<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            let output = self.output(left, &self.read(right)?, false)?;
            if matches!(self.mode, Mode::PendingValues) {
                std::future::pending::<()>().await;
            }
            Ok(output)
        })
    }
    fn elem_max<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        right: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if matches!(self.mode, Mode::UnsupportedPair) {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "pair extrema unavailable",
                )
                .into());
            }
            let output = self.output(left, &self.read(right)?, true)?;
            if matches!(self.mode, Mode::PendingValues) {
                std::future::pending::<()>().await;
            }
            Ok(output)
        })
    }
    fn elem_le<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async { panic!("comparison masks are not extrema origins") })
    }
    fn elem_ge<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async { panic!("comparison masks are not extrema origins") })
    }
    fn scalar_min(&self, input: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        self.output(input, &vec![scalar; input.shape().iter().product()], false)
    }
    fn scalar_max(&self, input: &GpuTensorHandle, scalar: f64) -> anyhow::Result<GpuTensorHandle> {
        self.output(input, &vec![scalar; input.shape().iter().product()], true)
    }
    fn fill_like(
        &self,
        input: &GpuTensorHandle,
        scalar: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        self.fills.fetch_add(1, Ordering::SeqCst);
        if matches!(self.mode, Mode::UnsupportedFill) {
            return Ok(None);
        }
        if matches!(self.mode, Mode::AliasFill) {
            return Ok(Some(input.clone()));
        }
        let shape = if matches!(self.mode, Mode::WrongFill) {
            vec![1, 1]
        } else {
            input.shape().to_vec()
        };
        Ok(Some(
            self.allocate(vec![scalar; shape.iter().product()], shape)
                .handle,
        ))
    }
    fn repmat(
        &self,
        input: &GpuTensorHandle,
        repetitions: &[usize],
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        let result = (|| -> anyhow::Result<GpuTensorHandle> {
            self.repeats.fetch_add(1, Ordering::SeqCst);
            if matches!(self.mode, Mode::WrongRepmat) {
                return Ok(input.clone());
            }
            let data = self.read(input)?;
            assert_eq!(input.shape().len(), 2);
            let rows = input.shape()[0];
            let columns = input.shape()[1];
            let shape = vec![rows * repetitions[0], columns * repetitions[1]];
            let mut output = Vec::new();
            for column in 0..shape[1] {
                for row in 0..shape[0] {
                    output.push(data[(column % columns) * rows + row % rows]);
                }
            }
            Ok(self.allocate(output, shape).handle)
        })();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
}
pub(super) fn provider(mode: Mode) -> &'static Provider {
    Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        records: Mutex::new(Vec::new()),
        mode,
        values: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
        fills: AtomicUsize::new(0),
        repeats: AtomicUsize::new(0),
    }))
}
pub(super) fn all_retired_except(provider: &Provider, retained: &[u64]) {
    for (id, probe) in provider.records.lock().unwrap().iter() {
        assert_eq!(
            probe.retirements(),
            usize::from(!retained.contains(id)),
            "allocation {id}"
        );
    }
}
