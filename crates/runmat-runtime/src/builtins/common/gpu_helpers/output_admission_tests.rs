//! Shared output admission must reject unavailable allocations, not just descriptors.
use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, HostTensorView, ThreadProviderGuard,
};

struct Provider(TestAllocationOwner);

impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.0.device_id()
    }

    fn device_info(&self) -> String {
        "output admission fixture".into()
    }

    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("output admission fixture does not upload")
    }

    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("output admission fixture does not download") })
    }
}

#[test]
fn unary_and_binary_admission_reject_unavailable_output_leases() {
    let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let allocate = || {
        provider
            .0
            .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap()
    };
    let left = allocate();
    let right = allocate();
    let output = allocate();
    let unary = UnaryGpuOutputContract {
        storage: GpuTensorStorage::Real,
        precision: Some(ProviderPrecision::F32),
        integer: None,
        logical: false,
        alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
    };
    let binary = BinaryGpuOutputContract {
        shape: vec![1, 1],
        storage: unary.storage,
        precision: unary.precision,
        integer: unary.integer,
        logical: unary.logical,
        alias: unary.alias,
    };
    assert!(unary_gpu_output_matches(
        &output.handle,
        &left.handle,
        provider,
        unary
    ));
    assert!(binary_gpu_output_matches(
        &output.handle,
        &left.handle,
        &right.handle,
        provider,
        &binary,
    ));

    output.probe.mark_unavailable();
    assert!(!unary_gpu_output_matches(
        &output.handle,
        &left.handle,
        provider,
        unary
    ));
    assert!(!binary_gpu_output_matches(
        &output.handle,
        &left.handle,
        &right.handle,
        provider,
        &binary,
    ));
    left.handle.allocation_lease().validate_live().unwrap();
    right.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(output.probe.retirements(), 0);
}
