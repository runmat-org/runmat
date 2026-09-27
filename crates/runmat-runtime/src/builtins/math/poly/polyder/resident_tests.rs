use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProviderFuture, HostTensorView,
    NumericElementType,
};
use std::future::Future;
use std::task::{Context, Poll};

enum Mode {
    Pending,
    Failure,
    Alias,
}
struct Provider {
    owner: TestAllocationOwner,
    mode: Mode,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned derivative fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("unexpected download")
    }
    fn polyder_single<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            match self.mode {
                Mode::Pending => futures::future::pending().await,
                Mode::Failure => anyhow::bail!("injected derivative failure"),
                Mode::Alias => Ok(input.clone()),
            }
        })
    }
}

#[test]
fn pending_derivative_retains_input_until_cancel_and_failure_never_downloads() {
    let _lock = crate::builtins::common::test_support::accel_test_lock();
    for mode in [Mode::Pending, Mode::Failure, Mode::Alias] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            mode,
        }));
        let _scope = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let allocation = provider
            .owner
            .allocate_zeroed(vec![1, 3], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let value = Value::GpuTensor(allocation.handle);
        let mut future = Box::pin(async move { try_gpu_derivative_single(&value).await });
        let waker = futures::task::noop_waker();
        let result = future.as_mut().poll(&mut Context::from_waker(&waker));
        match provider.mode {
            Mode::Pending => assert!(result.is_pending()),
            Mode::Failure => {
                let Poll::Ready(Err(error)) = result else {
                    panic!("execution error required")
                };
                assert!(error.message().contains("injected derivative failure"));
            }
            Mode::Alias => {
                let Poll::Ready(Err(error)) = result else {
                    panic!("alias error required")
                };
                assert!(error.message().contains("invalid provider output"));
            }
        }
        if matches!(provider.mode, Mode::Pending) {
            assert_eq!(allocation.probe.retirements(), 0);
        }
        drop(future);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}
