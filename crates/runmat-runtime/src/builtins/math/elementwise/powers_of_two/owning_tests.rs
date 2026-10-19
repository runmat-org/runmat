use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use runmat_value::Value;

struct Provider {
    owner: TestAllocationOwner,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "powers of two alias fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no download after invalid provider output") })
    }
    fn unary_pow2<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(input.with_view(input.view().clone())?) })
    }
    fn unary_nextpow2<'a>(
        &'a self,
        input: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { Ok(input.with_view(input.view().clone())?) })
    }
}

#[test]
fn powers_of_two_reject_new_view_alias_without_retiring_input() {
    for name in ["pow2", "nextpow2"] {
        let provider = Box::leak(Box::new(Provider {
            owner: TestAllocationOwner::new(),
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
        assert!(error.message().contains("malformed"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert_eq!(input.probe.retirements(), 0);
        input.handle.allocation_lease().validate_live().unwrap();
    }
}
