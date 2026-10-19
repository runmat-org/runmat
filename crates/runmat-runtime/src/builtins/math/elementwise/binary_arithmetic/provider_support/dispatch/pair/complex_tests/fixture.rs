use super::*;
use runmat_accelerate_api::test_support::{RetirementProbe, TestAllocationOwner};
use runmat_accelerate_api::{AccelDownloadFuture, GpuHandleProvenance, HostTensorView};
use std::sync::atomic::AtomicUsize;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug)]
pub(super) enum ResultMode {
    Valid,
    WrongStorage,
    WrongPrecision,
    WrongShape,
    ForeignOwner,
    BorrowedAlias,
    Unsupported,
    Failed,
    Pending,
}

pub(super) struct ComplexProvider {
    pub(super) owner: TestAllocationOwner,
    foreign: TestAllocationOwner,
    mode: ResultMode,
    pub(super) calls: AtomicUsize,
    pub(super) output: Mutex<Option<RetirementProbe>>,
}

impl AccelProvider for ComplexProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }

    fn device_info(&self) -> String {
        "complex arithmetic admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unexpected host fallback") })
    }

    fn elem_add<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if matches!(self.mode, ResultMode::Pending) {
                return futures::future::pending().await;
            }
            if matches!(self.mode, ResultMode::Unsupported) {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "complex addition unsupported",
                )
                .into());
            }
            if matches!(self.mode, ResultMode::Failed) {
                return Err(anyhow::anyhow!("complex addition device lost").into());
            }
            if matches!(self.mode, ResultMode::BorrowedAlias) {
                return Ok(left.clone());
            }
            let owner = if matches!(self.mode, ResultMode::ForeignOwner) {
                &self.foreign
            } else {
                &self.owner
            };
            let storage = if matches!(self.mode, ResultMode::WrongStorage) {
                GpuTensorStorage::Real
            } else {
                GpuTensorStorage::ComplexInterleaved
            };
            let element = if matches!(self.mode, ResultMode::WrongPrecision) {
                NumericElementType::F64
            } else {
                NumericElementType::F32
            };
            let shape = if matches!(self.mode, ResultMode::WrongShape) {
                vec![1, 2]
            } else {
                vec![2, 2]
            };
            let allocation = owner.allocate_zeroed(shape, element, storage)?;
            *self.output.lock().unwrap() = Some(allocation.probe);
            Ok(allocation.handle)
        })
    }
}

pub(super) fn with_case(mode: ResultMode, check: impl FnOnce(&'static ComplexProvider)) {
    let _guard = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static ComplexProvider = Box::leak(Box::new(ComplexProvider {
        owner: TestAllocationOwner::new(),
        foreign: TestAllocationOwner::new(),
        mode,
        calls: AtomicUsize::new(0),
        output: Mutex::new(None),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    check(provider);
}

pub(super) fn inputs(provider: &ComplexProvider) -> (GpuTensorHandle, GpuTensorHandle) {
    let (left_shape, right_shape) = if matches!(provider.mode, ResultMode::BorrowedAlias) {
        (vec![2, 2], vec![2, 2])
    } else {
        (vec![2, 1], vec![1, 2])
    };
    let left = provider
        .owner
        .allocate_zeroed(
            left_shape,
            NumericElementType::F32,
            GpuTensorStorage::ComplexInterleaved,
        )
        .unwrap()
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let right = provider
        .owner
        .allocate_zeroed(right_shape, NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap()
        .handle;
    (left, right)
}
