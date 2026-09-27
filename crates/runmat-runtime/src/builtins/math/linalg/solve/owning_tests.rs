use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType, ProviderInvOptions,
    ProviderNormOrder, ThreadProviderGuard,
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
        scalar: bool,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move {
            if self.alias {
                let view = if scalar {
                    input.view().clone().with_shape(vec![1, 1])
                } else {
                    input.view().clone()
                };
                Ok(input.with_view(view)?)
            } else {
                Err(
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "native linalg failed")
                        .into(),
                )
            }
        })
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "native linalg ownership witness".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no fallback after native failure") })
    }
    fn inv<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: ProviderInvOptions,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.result(input, false)
    }
    fn norm<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
        _: ProviderNormOrder,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        self.result(input, true)
    }
}

#[test]
fn inverse_and_norm_keep_provider_causes_and_borrowed_inputs_on_rejection() {
    for name in ["inv", "norm"] {
        for alias in [false, true] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                alias,
            }));
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let error = futures::executor::block_on(crate::call_builtin_async(
                name,
                &[Value::GpuTensor(input.handle.clone())],
            ))
            .unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            if !alias {
                assert!(error
                    .source
                    .as_ref()
                    .unwrap()
                    .downcast_ref::<std::io::Error>()
                    .is_some());
            }
            assert_eq!(input.probe.retirements(), 0);
            input.handle.allocation_lease().validate_live().unwrap();
        }
    }
}
