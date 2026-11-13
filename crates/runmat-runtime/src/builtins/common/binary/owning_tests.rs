use super::*;
use crate::builtins::common::gpu_helpers::{BinaryGpuOutputContract, GpuOutputAliasPolicy};
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, GpuTensorStorage, HostTensorView,
    NumericElementType, ProviderPrecision, ThreadProviderGuard,
};

struct Provider(TestAllocationOwner);
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
    fn device_info(&self) -> String {
        "binary owning validation fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused download") })
    }
}

#[test]
fn binary_validation_consumes_rejection_and_publishes_only_valid_owning_result() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocate = |shape| {
        provider
            .0
            .allocate_zeroed(shape, NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap()
    };
    let left = allocate(vec![2, 1]);
    let right = allocate(vec![2, 1]);
    let contract = BinaryGpuOutputContract {
        shape: vec![2, 1],
        storage: GpuTensorStorage::Real,
        precision: Some(ProviderPrecision::F64),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    let malformed = allocate(vec![1, 1]);
    let error = validate_resident_output(
        provider,
        &left.handle,
        &right.handle,
        malformed.handle,
        &contract,
    )
    .unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    assert_eq!(malformed.probe.retirements(), 1);
    let alias = left.handle.with_view(left.handle.view().clone()).unwrap();
    assert!(
        validate_resident_output(provider, &left.handle, &right.handle, alias, &contract).is_err()
    );
    assert_eq!(left.probe.retirements(), 0);
    let valid = allocate(vec![2, 1]);
    let published = validate_resident_output(
        provider,
        &left.handle,
        &right.handle,
        valid.handle,
        &contract,
    )
    .unwrap();
    assert_eq!(valid.probe.retirements(), 0);
    let external = published.clone();
    drop(published);
    assert_eq!(valid.probe.retirements(), 0);
    drop(external);
    assert_eq!(valid.probe.retirements(), 1);
    left.handle.allocation_lease().validate_live().unwrap();
    right.handle.allocation_lease().validate_live().unwrap();
}
