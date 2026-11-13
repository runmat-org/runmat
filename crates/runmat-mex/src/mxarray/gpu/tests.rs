use super::*;
use runmat_accelerate_api::{test_support::TestAllocationOwner, NumericElementType};

#[test]
fn published_views_and_input_clones_keep_the_actual_allocation_alive() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 3], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let original = allocation.handle.clone();
    let array = MxGpuLease::borrowed(allocation.handle);
    let published = array.publish(&[3, 2]).unwrap();
    assert!(published.shares_allocation_with(&original));
    assert_ne!(published.view_identity(), original.view_identity());
    assert_eq!(original.shape(), [2, 3]);
    drop(array);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(published);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(original);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn invalid_publication_does_not_mutate_or_abandon_the_source() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![2, 3], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let array = MxGpuLease::owned(allocation.handle);
    assert!(array.publish(&[7]).is_err());
    assert_eq!(array.handle().shape(), [2, 3]);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(array);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn mx_gpu_class_cannot_relabel_an_owned_numeric_allocation_as_logical() {
    let owner = TestAllocationOwner::new();
    let allocation = owner
        .allocate_zeroed(vec![1, 1], NumericElementType::U8, GpuTensorStorage::Real)
        .unwrap();
    assert!(
        crate::MxArray::gpu_owned(MxClassId::Logical, MxGpuLease::owned(allocation.handle))
            .is_err()
    );
    assert_eq!(allocation.probe.retirements(), 1);
}
