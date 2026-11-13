use super::*;
use runmat_accelerate_api::{
    test_support::{RetirementProbe, TestAllocationOwner},
    AccelDownloadFuture, AccelProvider, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use std::future::Future;
use std::sync::Mutex;
use std::task::{Context, Poll};

struct Provider {
    owner: TestAllocationOwner,
    intermediate: Mutex<Option<RetirementProbe>>,
    fail: bool,
}

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "ctranspose ownership fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected fallback upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no fallback download on operational error or pending hook") })
    }
    fn transpose(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        let mut shape = input.shape().to_vec();
        shape.swap(0, 1);
        let allocation = self.owner.allocate_zeroed(
            shape,
            NumericElementType::F64,
            runmat_accelerate_api::handle_storage(input),
        )?;
        *self.intermediate.lock().unwrap() = Some(allocation.probe);
        Ok(allocation.handle)
    }
    fn unary_conj<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, Option<GpuTensorHandle>> {
        Box::pin(async move {
            if self.fail {
                return Err(
                    runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                        "conjugate execution failed"
                    ))
                    .with_unproven_effects(),
                );
            }
            futures::future::pending().await
        })
    }
}

#[test]
fn ctranspose_owns_intermediate_across_pending_and_operational_failure() {
    for fail in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            intermediate: Mutex::new(None),
            fail,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(
                vec![2, 3],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let mut invocation = Box::pin(ctranspose_gpu(input.handle.clone()));
        let waker = futures::task::noop_waker();
        let poll = invocation.as_mut().poll(&mut Context::from_waker(&waker));
        if fail {
            let Poll::Ready(Err(error)) = poll else {
                panic!("expected actual hook failure");
            };
            assert!(error.message().contains("conjugate execution failed"));
            assert!(error.source.is_some());
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        } else {
            assert!(poll.is_pending());
            assert_eq!(
                provider
                    .intermediate
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .retirements(),
                0
            );
        }
        drop(invocation);
        assert_eq!(
            provider
                .intermediate
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .retirements(),
            1
        );
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
