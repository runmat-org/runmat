use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProviderFuture, GpuTensorStorage, HostTensorView, NumericElementType,
};
use std::{
    future::Future,
    sync::Mutex,
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
enum Mode {
    Pending,
    Unsupported,
    Alias,
}
struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
    expanded: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "binary owning expansion fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("no host fallback in expansion test") })
    }
    fn repmat(&self, handle: &GpuTensorHandle, reps: &[usize]) -> anyhow::Result<GpuTensorHandle> {
        let shape = (0..handle.shape().len().max(reps.len()))
            .map(|i| {
                handle.shape().get(i).copied().unwrap_or(1) * reps.get(i).copied().unwrap_or(1)
            })
            .collect();
        let allocation =
            self.owner
                .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)?;
        self.expanded.lock().unwrap().push(allocation.probe);
        Ok(allocation.handle)
    }
    fn elem_add<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.mode {
                Mode::Pending => futures::future::pending().await,
                Mode::Unsupported => Err(runmat_accelerate_api::unsupported_provider_operation(
                    "fixture add unsupported",
                )),
                Mode::Alias => Ok(left.with_view(left.view().clone())?),
            }
        })
    }
}

#[test]
fn expanded_owned_operands_retire_on_cancel_unsupported_and_rejected_alias() {
    for mode in [Mode::Pending, Mode::Unsupported, Mode::Alias] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            mode,
            expanded: Mutex::new(Vec::new()),
        }));
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let left = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let right = provider
            .owner
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let mut future = Box::pin(try_broadcast(
            ArithmeticProviderOperation::Add,
            provider,
            &left.handle,
            &right.handle,
        ));
        let waker = futures::task::noop_waker();
        match (mode, future.as_mut().poll(&mut Context::from_waker(&waker))) {
            (Mode::Pending, Poll::Pending) => {}
            (Mode::Unsupported, Poll::Ready(Ok(None))) => {}
            (Mode::Alias, Poll::Ready(Err(error))) => {
                assert!(error.message().contains("invalid element-wise output"))
            }
            (_, result) => panic!("unexpected production expansion result: {result:?}"),
        }
        drop(future);
        let expanded = provider.expanded.lock().unwrap();
        assert_eq!(expanded.len(), 2);
        assert!(expanded.iter().all(|probe| probe.retirements() == 1));
        assert_eq!(left.probe.retirements(), 0);
        assert_eq!(right.probe.retirements(), 0);
        left.handle.allocation_lease().validate_live().unwrap();
        right.handle.allocation_lease().validate_live().unwrap();
    }
}
