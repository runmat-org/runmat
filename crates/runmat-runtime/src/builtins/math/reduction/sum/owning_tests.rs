use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProviderFuture, GpuTensorStorage, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};
struct Provider {
    owner: TestAllocationOwner,
    cleaned: Mutex<Vec<RetirementProbe>>,
    fail: bool,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "sum cleaned-input lifecycle".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("execution failure must not gather") })
    }
    fn map_nan_to_zero(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        let result = self.owner.allocate_zeroed(
            input.shape().to_vec(),
            NumericElementType::F64,
            GpuTensorStorage::Real,
        )?;
        self.cleaned.lock().unwrap().push(result.probe);
        Ok(result.handle)
    }
    fn reduce_sum_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.fail {
                Err(anyhow::Error::new(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "sum reduction failed",
                )))
            } else {
                futures::future::pending().await
            }
        })
    }
}

#[test]
fn omitnan_sum_retires_cleaned_owner_on_cancel_and_failure_without_gathering() {
    for fail in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            cleaned: Mutex::new(Vec::new()),
            fail,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let parsed = ParsedArguments {
            selection: DimSelection::Dim(1),
            nan_mode: ReductionNaN::Omit,
            output: OutputTemplate::Double,
        };
        let mut future = Box::pin(sum_gpu_with_omitnan(input.handle.clone(), &parsed));
        let waker = futures::task::noop_waker();
        match (fail, future.as_mut().poll(&mut Context::from_waker(&waker))) {
            (false, Poll::Pending) => {}
            (true, Poll::Ready(Err(error))) => {
                assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(error
                    .source
                    .as_ref()
                    .unwrap()
                    .downcast_ref::<std::io::Error>()
                    .is_some());
            }
            (_, outcome) => panic!("unexpected production sum outcome: {outcome:?}"),
        }
        drop(future);
        let cleaned = provider.cleaned.lock().unwrap();
        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
