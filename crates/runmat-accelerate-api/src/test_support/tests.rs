use super::*;

#[test]
fn unavailable_allocation_keeps_borrowed_payload_until_final_owner_drops() {
    let factory = TestAllocationOwner::new();
    let allocation = factory
        .allocate_zeroed(vec![4, 2], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let other = factory
        .allocate_zeroed(vec![1, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .unwrap();
    let alias = allocation.handle.clone();
    let weak = allocation.handle.downgrade();
    allocation.probe.mark_unavailable();
    allocation.probe.mark_unavailable(); // terminal and idempotent, not another free
    assert!(matches!(
        alias.allocation_lease().validate_live(),
        Err(AllocationAcquisitionError::NotLive)
    ));
    assert!(matches!(
        weak.acquire(),
        Err(AllocationAcquisitionError::NotLive)
    ));
    assert!(allocation.probe.with_data(&alias, |_| ()).is_err());
    assert!(other.handle.allocation_lease().validate_live().is_ok());
    assert_eq!(allocation.probe.retirements(), 0);
    assert!(allocation.probe.owner.record.lock().unwrap().0.is_some());
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 0);
    assert!(allocation.probe.owner.record.lock().unwrap().0.is_some());
    drop(alias);
    assert_eq!(allocation.probe.retirements(), 1);
    assert!(allocation.probe.owner.record.lock().unwrap().0.is_none());
    drop(other.handle);
    assert_eq!(other.probe.retirements(), 1);
}

#[test]
fn injected_retirement_failure_retains_payload_without_reacquisition_or_retry() {
    let allocation = TestAllocationOwner::new()
        .allocate_zeroed(vec![2], NumericElementType::U64, GpuTensorStorage::Real)
        .unwrap();
    allocation.probe.fail_retirement("fixture retained payload");
    let weak = allocation.handle.downgrade();
    let error = allocation.handle.release().unwrap_err();
    assert_eq!(allocation.probe.retirements(), 1);
    assert!(matches!(
        weak.acquire(),
        Err(AllocationAcquisitionError::NotLive)
    ));
    assert!(allocation.probe.owner.record.lock().unwrap().0.is_some());
    assert!(matches!(
        allocation
            .probe
            .owner
            .validate_live(allocation.probe.owner.identity),
        Err(AllocationAcquisitionError::NotLive)
    ));
    drop(error);
    assert_eq!(allocation.probe.retirements(), 1);
    assert!(allocation.probe.owner.record.lock().unwrap().0.is_some());
}

#[test]
fn typed_backing_is_moved_without_copy_and_native_address_uses_that_same_allocation() {
    let owner = TestAllocationOwner::new();
    let values = vec![1.5_f64, -2.0];
    let original = values.as_ptr() as u64;
    let allocation = owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(values),
            shape: vec![1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    allocation
        .probe
        .with_native_address(&allocation.handle, |address, bytes| {
            assert_eq!(address, original);
            assert_eq!(address % std::mem::align_of::<f64>() as u64, 0);
            assert_eq!(bytes, 16);
            // SAFETY: this callback holds the actual typed payload lock; no other
            // reader exists, the handle retains it, and the index is in bounds.
            unsafe {
                (address as *mut f64).add(1).write(7.25);
            }
        })
        .unwrap();
    allocation
        .probe
        .with_data(&allocation.handle, |data| {
            let HostNumericDataView::F64(data) = data else {
                panic!("lost native class")
            };
            assert_eq!(data, [1.5, 7.25]);
            assert_eq!(data.as_ptr() as u64, original);
        })
        .unwrap();
    let alias = allocation.handle.clone();
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(alias);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn typed_backing_preserves_complex_extent_logical_intent_and_empty_shape() {
    let owner = TestAllocationOwner::new();
    let complex = owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F32(vec![1.0, 2.0, 3.0, 4.0]),
            shape: vec![1, 1, 2],
            storage: GpuTensorStorage::ComplexInterleaved,
        })
        .unwrap();
    assert_eq!(complex.handle.physical().payload_bytes(), 16);
    complex
        .probe
        .with_native_address(&complex.handle, |address, bytes| {
            assert_ne!(address, 0);
            assert_eq!(bytes, 16);
        })
        .unwrap();
    let logical = owner.allocate_logical(vec![0, 1], vec![2, 1]).unwrap();
    assert!(handle_is_logical(&logical.handle));
    assert_eq!(
        logical.handle.physical().element_type(),
        NumericElementType::U8
    );
    assert!(owner.allocate_logical(vec![2], vec![1, 1]).is_err());
    let empty = owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::I64(vec![]),
            shape: vec![2, 0, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert_eq!(empty.handle.shape(), [2, 0, 3]);
    empty
        .probe
        .with_native_address(&empty.handle, |address, bytes| {
            assert_eq!((address, bytes), (0, 0));
        })
        .unwrap();
    assert!(logical.probe.with_data(&complex.handle, |_| ()).is_err());
    assert!(owner
        .allocate_numeric(HostNumericTensorOwned {
            data: HostNumericDataOwned::F64(vec![1.0]),
            shape: vec![2],
            storage: GpuTensorStorage::Real,
        })
        .is_err());
}
