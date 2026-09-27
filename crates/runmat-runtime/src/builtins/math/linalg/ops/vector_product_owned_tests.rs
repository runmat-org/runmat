use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    GpuTensorHandle, GpuTensorStorage, HostTensorView, NumericElementType, ThreadProviderGuard,
};
use runmat_value::Value;

struct Provider {
    owner: TestAllocationOwner,
    fail: bool,
}
impl Provider {
    fn result(&self, input: &GpuTensorHandle) -> anyhow::Result<GpuTensorHandle> {
        if self.fail {
            anyhow::bail!("vector product execution failure");
        }
        Ok(input.with_view(input.view().clone())?)
    }
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "vector product owning fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no fallback for malformed output or operational failure") })
    }
    fn cross(
        &self,
        left: &GpuTensorHandle,
        _: &GpuTensorHandle,
        _: Option<usize>,
    ) -> anyhow::Result<GpuTensorHandle> {
        self.result(left)
    }
    fn dot<'a>(
        &'a self,
        left: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
        _: Option<usize>,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.result(left) })
    }
}

#[test]
fn vector_product_rejection_preserves_borrowed_allocations_without_fallback() {
    for name in ["cross", "dot"] {
        for fail in [false, true] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                fail,
            }));
            let _selected = ThreadProviderGuard::set(Some(provider));
            let left = provider
                .owner
                .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let right = provider
                .owner
                .allocate_zeroed(vec![3, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let error = futures::executor::block_on(crate::call_builtin_async(
                name,
                &[
                    Value::GpuTensor(left.handle.clone()),
                    Value::GpuTensor(right.handle.clone()),
                ],
            ))
            .unwrap_err();
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            if fail {
                assert!(error.source.is_some());
            }
            left.handle.allocation_lease().validate_live().unwrap();
            right.handle.allocation_lease().validate_live().unwrap();
            assert_eq!(left.probe.retirements(), 0);
            assert_eq!(right.probe.retirements(), 0);
            drop(left.handle);
            drop(right.handle);
            assert_eq!(left.probe.retirements(), 1);
            assert_eq!(right.probe.retirements(), 1);
        }
    }
}
