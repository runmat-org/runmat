use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, AccelProviderFuture,
    HostTensorView, NumericElementType, ThreadProviderGuard,
};

struct Provider(TestAllocationOwner);
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "logarithm ownership fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unexpected upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { panic!("no download after probe failure") })
    }
    fn reduce_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> AccelProviderFuture<'a, GpuTensorHandle> {
        Box::pin(async {
            Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "probe failed").into())
        })
    }
}

#[test]
fn logarithm_probe_keeps_original_source_and_rejected_view_keeps_input_alive() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error =
        futures::executor::block_on(evaluate(LogarithmOperation::Natural, input.handle.clone()))
            .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let probe = error.source.as_ref().unwrap();
    assert!(probe
        .source()
        .unwrap()
        .downcast_ref::<std::io::Error>()
        .is_some());
    let alias = input.handle.with_view(input.handle.view().clone()).unwrap();
    let error = contract::validate(LogarithmOperation::Natural, provider, &input.handle, alias)
        .unwrap_err();
    assert!(error.message().contains("malformed"));
    assert_eq!(input.probe.retirements(), 0);
    input.handle.allocation_lease().validate_live().unwrap();
}
