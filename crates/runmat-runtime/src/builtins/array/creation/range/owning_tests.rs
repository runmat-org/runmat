use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, AccelProviderFuture, GpuTensorStorage, HostTensorView,
    NumericElementType,
};
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

struct ReductionOwner {
    owner: TestAllocationOwner,
    outputs: Mutex<Vec<RetirementProbe>>,
    alias: bool,
    pending: bool,
}

impl AccelProvider for ReductionOwner {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned range reduction fixture".into()
    }
    fn upload(&self, _: &HostTensorView<'_>) -> anyhow::Result<GpuTensorHandle> {
        unreachable!("no uploads in reduction lifetime witness")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        unreachable!("no downloads in reduction lifetime witness")
    }
    fn reduce_min<'a>(
        &'a self,
        source: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.alias {
                return Ok(source.clone());
            }
            let allocation = self.owner.allocate_zeroed(
                vec![1, 1],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.outputs.lock().unwrap().push(allocation.probe);
            Ok(allocation.handle)
        })
    }
    fn reduce_max<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.pending {
                futures::future::pending::<()>().await;
            }
            Err(anyhow::anyhow!("injected second reduction failure"))
        })
    }
}

#[test]
fn range_intermediate_owns_its_lifetime_across_failure_and_pending_drop() {
    for alias in [false, true] {
        for pending in [false, true] {
            let provider = Box::leak(Box::new(ReductionOwner {
                owner: TestAllocationOwner::new(),
                outputs: Mutex::new(Vec::new()),
                alias,
                pending,
            }));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let mut future = Box::pin(range_gpu_all(provider, &input.handle));
            let waker = futures::task::noop_waker();
            let mut context = Context::from_waker(&waker);
            match future.as_mut().poll(&mut context) {
                Poll::Pending => assert!(pending),
                Poll::Ready(output) => {
                    assert!(!pending);
                    assert!(output.is_none());
                }
            }
            assert_eq!(input.probe.retirements(), 0);
            if !alias && pending {
                assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 0);
            }
            drop(future);
            if !alias {
                assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
            }
            assert_eq!(input.probe.retirements(), 0);
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}
