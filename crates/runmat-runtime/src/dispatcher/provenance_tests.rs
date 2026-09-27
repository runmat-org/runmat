use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, GpuHandleProvenance, GpuTensorStorage, NumericElementType,
};

#[test]
fn result_projection_preserves_source_view_and_allocation_lifetime() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 1], NumericElementType::U64, GpuTensorStorage::Real)
        .unwrap();
    let source = allocation
        .handle
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let mut result = Value::GpuTensor(allocation.handle.clone());
    propagate_gpu_provenance(&[Value::GpuTensor(source)], &mut result).unwrap();
    let Value::GpuTensor(projected) = result else {
        unreachable!()
    };
    assert!(projected.shares_allocation_with(&allocation.handle));
    assert_eq!(projected.provenance(), Some(GpuHandleProvenance::Explicit));
    assert_eq!(allocation.handle.provenance(), None);
    assert_eq!(projected.shape(), &[2, 1]);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 0);
    projected.allocation_lease().validate_live().unwrap();
    drop(projected);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn explicit_producer_intent_survives_without_resident_arguments() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![0, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let mut result = Value::GpuTensor(
        allocation
            .handle
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap(),
    );
    propagate_gpu_provenance(&[], &mut result).unwrap();
    let Value::GpuTensor(projected) = result else {
        unreachable!()
    };
    assert_eq!(projected.provenance(), Some(GpuHandleProvenance::Explicit));
    assert_eq!(projected.shape(), &[0, 3]);
}
