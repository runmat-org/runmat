use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use runmat_value::Value;

struct Provider {
    owner: TestAllocationOwner,
    alias: bool,
}
impl Provider {
    fn result<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.alias {
                Ok(input.with_view(input.view().clone())?)
            } else {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "unary failed").into())
            }
        })
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "magnitude/phase/sign ownership witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no fallback after provider failure") })
    }
    fn unary_abs<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.result(input)
    }
    fn unary_angle<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.result(input)
    }
    fn unary_sign<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.result(input)
    }
}

#[test]
fn unary_dispatch_preserves_provider_failure_and_rejects_physical_alias() {
    for name in ["abs", "angle", "sign"] {
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
            let error = futures::executor::block_on(crate::call_builtin_async(
                name,
                &[Value::GpuTensor(input.handle.clone())],
            ))
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
                assert_eq!(cause.to_string(), "unary failed");
            }
            assert_eq!(input.probe.retirements(), 0);
            input.handle.allocation_lease().validate_live().unwrap();
        }
    }
}
