//! Real fixture-lease admission at the canonical final publication boundary.
//! This is not a compound execution, GPU math, or hardware qualification receipt.
use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuHandleProvenance, GpuTensorStorage, NumericElementType,
};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn completed_publication_keeps_source_provenance_without_retiring_borrowed_input() {
    for provenance in [
        GpuHandleProvenance::Automatic,
        GpuHandleProvenance::Explicit,
    ] {
        let owner = TestAllocationOwner::new();
        let source = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let input = source.handle.clone().with_provenance(provenance).unwrap();
        let output = owner
            .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
            .unwrap();
        let Value::GpuTensor(published) = publish_completed(output.handle, &input).unwrap() else {
            panic!("ordinary resident publication");
        };
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&published),
            Some(provenance)
        );
        assert_eq!(published.physical().element_type(), NumericElementType::F32);
        drop(published);
        assert_eq!(output.probe.retirements(), 1);
        assert_eq!(source.probe.retirements(), 0);
        input.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn quarantined_completed_output_publication_is_terminal_with_original_adoption_source() {
    let owner = TestAllocationOwner::new();
    let source = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let input = source
        .handle
        .clone()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let output = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    output.probe.mark_unavailable();
    let error = publish_completed(output.handle, &input).unwrap_err();
    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let retained = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
        .unwrap();
    assert!(retained.is_after_effect());
    let original = std::error::Error::source(retained)
        .unwrap()
        .downcast_ref::<crate::RuntimeError>()
        .unwrap();
    assert!(std::error::Error::source(original).is_some());
    assert_eq!(source.probe.retirements(), 0);
    input.allocation_lease().validate_live().unwrap();
}
