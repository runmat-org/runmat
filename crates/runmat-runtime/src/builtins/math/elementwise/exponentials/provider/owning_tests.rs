use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    HostTensorView, NumericElementType, ThreadProviderGuard,
};

struct Provider {
    owner: TestAllocationOwner,
    alias: bool,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "exponential ownership fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no fallback after execution failure") })
    }
    fn unary_exp<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.alias {
                Ok(input.with_view(input.view().clone())?)
            } else {
                Err(
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "exponential failed")
                        .into(),
                )
            }
        })
    }
}

#[test]
fn exponential_failure_and_new_view_alias_preserve_borrowed_allocation() {
    for alias in [false, true] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
            alias,
        }));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let error =
            futures::executor::block_on(evaluate(ExponentialOperation::Exp, input.handle.clone()))
                .unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        if alias {
            assert!(error.message().contains("malformed"));
        } else {
            let source = error.source.as_ref().expect("provider source retained");
            let provider = source
                .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                .expect("typed provider execution carrier retained");
            let cause = std::error::Error::source(provider)
                .expect("original provider cause retained")
                .downcast_ref::<std::io::Error>()
                .expect("original I/O error remains downcastable");
            assert_eq!(cause.kind(), std::io::ErrorKind::BrokenPipe);
            assert_eq!(cause.to_string(), "exponential failed");
        }
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
