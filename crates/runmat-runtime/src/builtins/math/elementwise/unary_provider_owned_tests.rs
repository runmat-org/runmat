use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, NumericElementType, ThreadProviderGuard,
};

struct Provider(TestAllocationOwner);
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "unary ownership fixture".into()
    }
    fn upload(&self, _: &runmat_accelerate_api::HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("validation does not upload")
    }
    fn download<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        Box::pin(async { panic!("validation does not download") })
    }
}

#[test]
fn unary_validation_adopts_fresh_result_and_rejects_shared_physical_view() {
    let _serial = crate::builtins::common::test_support::accel_test_lock();
    let provider: &'static Provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .0
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let borrowed_view = input
        .handle
        .with_provenance(GpuHandleProvenance::Automatic)
        .unwrap();
    assert!(validate_real_unary_output(provider, &input.handle, borrowed_view, "sqrt").is_err());
    assert_eq!(input.probe.retirements(), 0);
    let fresh = provider
        .0
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let result = validate_real_unary_output(provider, &input.handle, fresh.handle, "sqrt").unwrap();
    assert_eq!(fresh.probe.retirements(), 0);
    drop(result);
    assert_eq!(fresh.probe.retirements(), 1);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[test]
fn unary_error_context_keeps_the_original_error_source() {
    let cause = crate::build_runtime_error("retained provider failure")
        .with_identifier("Provider:Failure")
        .build();
    let error = retained_internal("sqrt", cause);
    let cause = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<RuntimeError>()
        .unwrap();
    assert_eq!(cause.identifier(), Some("Provider:Failure"));
}
