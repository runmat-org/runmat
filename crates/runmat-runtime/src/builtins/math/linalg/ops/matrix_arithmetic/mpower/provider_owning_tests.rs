use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProviderFuture, HostTensorView, NumericElementType,
    ThreadProviderGuard,
};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    task::Context,
};

struct Provider {
    owner: TestAllocationOwner,
    calls: AtomicUsize,
    fail: bool,
    outputs: Mutex<Vec<RetirementProbe>>,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "matrix power ownership fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("unexpected download") })
    }
    fn matmul<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.calls.fetch_add(1, Ordering::SeqCst) != 0 {
                if self.fail {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "matmul failed",
                    )
                    .into());
                }
                return std::future::pending().await;
            }
            let output = self.owner.allocate_zeroed(
                vec![2, 2],
                NumericElementType::F64,
                GpuTensorStorage::Real,
            )?;
            self.outputs.lock().unwrap().push(output.probe);
            Ok(output.handle)
        })
    }
}

#[test]
fn repeated_power_releases_intermediate_on_pending_drop_and_execution_failure() {
    for fail in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            calls: AtomicUsize::new(0),
            fail,
            outputs: Mutex::new(Vec::new()),
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let mut future = Box::pin(binary_power(provider, &input.handle, 3));
        let waker = futures::task::noop_waker();
        let state = future.as_mut().poll(&mut Context::from_waker(&waker));
        if fail {
            let std::task::Poll::Ready(Err(error)) = state else {
                panic!("expected provider failure");
            };
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            let source = error.source.as_ref().expect("provider source retained");
            let provider = source
                .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                .expect("typed provider execution carrier retained");
            let cause = std::error::Error::source(provider)
                .expect("original provider cause retained")
                .downcast_ref::<std::io::Error>()
                .expect("original I/O error remains downcastable");
            assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
            assert_eq!(cause.to_string(), "matmul failed");
        } else {
            assert!(state.is_pending());
            assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 0);
        }
        drop(future);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(provider.outputs.lock().unwrap()[0].retirements(), 1);
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
