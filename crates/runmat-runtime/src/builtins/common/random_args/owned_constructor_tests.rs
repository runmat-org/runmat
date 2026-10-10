use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, AccelProvider, GpuTensorHandle,
    GpuTensorStorage, HostLogicalTensorView, HostTensorView, NumericElementType, ProviderPrecision,
};

struct Provider {
    owner: TestAllocationOwner,
}
impl AccelProvider for Provider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "owned constructor fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        anyhow::bail!("unused numeric upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async { anyhow::bail!("unused download") })
    }
    fn upload_logical_raw(
        &self,
        view: &HostLogicalTensorView,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        Ok(self
            .owner
            .allocate_logical(view.data.to_vec(), view.shape.to_vec())
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).with_unproven_effects()
            })?
            .handle)
    }
}

#[test]
fn constructors_require_logical_producer_intent_and_accept_native_u8() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let numeric = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::U8, GpuTensorStorage::Real)
        .unwrap();
    let rejected = validate_constructor_gpu_output(
        "logical-test",
        provider,
        numeric.handle,
        &[1, 2],
        GpuTensorStorage::Real,
        Some(ProviderPrecision::F64),
        None,
        true,
    );
    assert!(rejected.is_err());
    assert_eq!(numeric.probe.retirements(), 1);
    let logical = provider
        .owner
        .allocate_logical(vec![0, 1], vec![1, 2])
        .unwrap();
    let output = validate_constructor_gpu_output(
        "logical-test",
        provider,
        logical.handle,
        &[1, 2],
        GpuTensorStorage::Real,
        Some(ProviderPrecision::F64),
        None,
        true,
    )
    .unwrap();
    assert!(runmat_accelerate_api::handle_is_logical(&output));
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert_eq!(output.physical().element_type(), NumericElementType::U8);
    assert_eq!(logical.probe.retirements(), 0);
    drop(output);
    assert_eq!(logical.probe.retirements(), 1);
}

#[test]
fn fresh_constructor_rejects_distinct_view_of_protected_allocation() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let view = source
        .handle
        .with_view(source.handle.view().clone().with_shape(vec![2, 1]))
        .unwrap();
    assert_ne!(source.handle.view_identity(), view.view_identity());
    assert!(validate_constructor_gpu_output_protecting(
        "constructor-test",
        provider,
        view,
        &[2, 1],
        GpuTensorStorage::Real,
        Some(ProviderPrecision::F64),
        None,
        false,
        &[&source.handle]
    )
    .is_err());
    assert_eq!(source.probe.retirements(), 0);
    source.handle.allocation_lease().validate_live().unwrap();
    drop(source.handle);
    assert_eq!(source.probe.retirements(), 1);
}

#[test]
fn shared_resident_restoration_uses_logical_producer_without_relabeling() {
    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
    }));
    let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let source = provider
        .owner
        .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let explicit = source
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .unwrap();
    let output = crate::builtins::common::resident_output::restore_to_source_provider_protecting(
        Value::LogicalArray(runmat_value::LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap()),
        Some(&explicit),
        &[&source.handle],
    )
    .unwrap();
    let Value::GpuTensor(output) = output else {
        panic!("resident logical output required")
    };
    assert!(runmat_accelerate_api::handle_is_logical(&output));
    assert!(runmat_accelerate_api::handle_is_explicit(&output));
    assert_eq!(output.physical().element_type(), NumericElementType::U8);
    assert_eq!(output.shape(), &[1, 2]);
    assert!(!output.shares_allocation_with(&source.handle));
    assert!(!runmat_accelerate_api::handle_is_logical(&source.handle));
    assert_eq!(source.probe.retirements(), 0);
}
