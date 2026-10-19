use super::det_gpu_via_provider;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorHandle, GpuTensorStorage,
    HostTensorView, NumericElementType, ProviderExecutionError, ProviderLuResult,
    ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum Mode {
    Failure,
    PendingLu,
    PendingDownload,
}

struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    allocations: Mutex<Vec<RetirementProbe>>,
    calls: AtomicUsize,
    downloads: AtomicUsize,
}

impl Provider {
    fn allocate(&self, shape: Vec<usize>) -> anyhow::Result<GpuTensorHandle> {
        let allocation =
            self.owner
                .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)?;
        self.allocations.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "determinant failure and suspension witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("failed or cancelled determinant must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            assert!(matches!(self.mode, Mode::PendingDownload));
            self.downloads.fetch_add(1, Ordering::SeqCst);
            futures::future::pending().await
        })
    }
    fn lu<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelProviderFuture<'a, ProviderLuResult> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let combined = self.allocate(vec![2, 2])?;
            match self.mode {
                Mode::Failure => Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "determinant LU failed",
                )
                .into()),
                Mode::PendingLu => {
                    futures::future::pending::<()>().await;
                    drop(combined);
                    unreachable!("pending LU must be cancelled")
                }
                Mode::PendingDownload => Ok(ProviderLuResult {
                    combined,
                    lower: self.allocate(vec![2, 2])?,
                    upper: self.allocate(vec![2, 2])?,
                    perm_matrix: self.allocate(vec![2, 2])?,
                    perm_vector: self.allocate(vec![2, 1])?,
                }),
            }
        })
    }
}

#[test]
fn determinant_failure_and_cancellation_retire_owned_work_not_borrowed_input() {
    for mode in [Mode::Failure, Mode::PendingLu, Mode::PendingDownload] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            mode,
            allocations: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
            downloads: AtomicUsize::new(0),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let mut future = Box::pin(det_gpu_via_provider(provider, &input.handle));
        let waker = futures::task::noop_waker();
        match (mode, future.as_mut().poll(&mut Context::from_waker(&waker))) {
            (Mode::Failure, Poll::Ready(Err(error))) => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                let source = error
                    .source
                    .as_ref()
                    .unwrap()
                    .downcast_ref::<ProviderExecutionError>()
                    .expect("provider carrier retained");
                let cause = std::error::Error::source(source)
                    .unwrap()
                    .downcast_ref::<std::io::Error>()
                    .expect("original I/O cause retained");
                assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
                assert_eq!(cause.to_string(), "determinant LU failed");
            }
            (Mode::PendingLu | Mode::PendingDownload, Poll::Pending) => {
                assert!(provider
                    .allocations
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|probe| probe.retirements() == 0));
            }
            (_, actual) => panic!("unexpected determinant outcome: {actual:?}"),
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.downloads.load(Ordering::SeqCst),
            usize::from(matches!(mode, Mode::PendingDownload))
        );
        drop(future);
        let allocations = provider.allocations.lock().unwrap();
        assert_eq!(
            allocations.len(),
            if matches!(mode, Mode::PendingDownload) {
                5
            } else {
                1
            }
        );
        assert!(allocations.iter().all(|probe| probe.retirements() == 1));
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
        drop(input.handle);
        assert_eq!(input.probe.retirements(), 1);
    }
}
