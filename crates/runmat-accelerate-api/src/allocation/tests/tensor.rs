use super::*;

fn tensor(owner: &Arc<TestOwner>) -> GpuTensorHandle {
    GpuTensorHandle::from_allocation(
        AllocationLease::register(ID, owner.clone()).unwrap(),
        GpuTensorView::contiguous(vec![1, 2]),
    )
    .unwrap()
}

#[test]
fn clones_share_view_and_new_views_share_one_physical_retirement() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let original = tensor(&owner);
    let weak_view = original.downgrade();
    assert_eq!(
        weak_view.acquire().unwrap().view_identity(),
        original.view_identity()
    );
    let clone = original.clone();
    assert_eq!(clone.view_identity(), original.view_identity());
    assert_eq!(
        clone.shape().as_ptr(),
        original.shape().as_ptr(),
        "Clone does not copy shape storage"
    );
    let transposed =
        original
            .with_view(GpuTensorView::contiguous(vec![2, 1]).with_layout(
                GpuTensorLayout::Transpose(crate::TransposeInfo {
                    base_rows: 1,
                    base_cols: 2,
                }),
            ))
            .unwrap();
    assert_ne!(transposed.view_identity(), original.view_identity());
    assert!(transposed.shares_allocation_with(&original));
    assert_eq!(
        original.release().unwrap(),
        AllocationReleaseOutcome::ReferenceReleased
    );
    assert_eq!(
        clone.release().unwrap(),
        AllocationReleaseOutcome::ReferenceReleased
    );
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        transposed.release().unwrap(),
        AllocationReleaseOutcome::Retired
    );
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        weak_view.acquire().unwrap_err(),
        AllocationAcquisitionError::NotLive
    );
}

#[test]
fn wire_is_nonowning_and_acquisition_rejects_foreign_stale_or_forged_facts() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let original = tensor(&owner);
    let weak = owner.weak_lease(ID);
    let wire: GpuTensorWireDescriptor =
        serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();
    let acquired = weak.acquire_tensor(&wire).unwrap();
    assert!(acquired.shares_allocation_with(&original));
    let mut forged = wire.clone();
    forged.allocation =
        AllocationIdentity::new(ID.device_id() + 1, ID.allocation_id(), ID.generation());
    assert_eq!(
        weak.acquire_tensor(&forged).unwrap_err(),
        AllocationAcquisitionError::WrongOwner
    );
    forged.allocation =
        AllocationIdentity::new(ID.device_id(), ID.allocation_id(), ID.generation() + 1);
    assert_eq!(
        weak.acquire_tensor(&forged).unwrap_err(),
        AllocationAcquisitionError::StaleGeneration
    );
    forged = wire.clone();
    forged.physical = GpuPhysicalDescriptor::new(
        crate::NumericElementType::U16,
        crate::GpuTensorStorage::Real,
        1,
        2,
        2,
    )
    .unwrap();
    assert_eq!(
        weak.acquire_tensor(&forged).unwrap_err(),
        AllocationAcquisitionError::PhysicalMismatch
    );
    forged = wire.clone();
    forged.view = GpuTensorView::contiguous(vec![3]);
    assert!(matches!(
        weak.acquire_tensor(&forged),
        Err(AllocationAcquisitionError::InvalidView(
            GpuViewError::OutsideAllocation
        ))
    ));
    drop(original);
    drop(acquired);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        weak.acquire_tensor(&wire).unwrap_err(),
        AllocationAcquisitionError::NotLive
    );
}

#[test]
fn smaller_native_views_validate_without_expanding_physical_capacity() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let original = tensor(&owner);
    let smaller = original
        .with_view(GpuTensorView::contiguous(vec![1]))
        .unwrap();
    assert_eq!(smaller.physical(), original.physical());
    assert_eq!(smaller.shape(), &[1]);
    assert!(original
        .with_view(GpuTensorView::contiguous(vec![1, 3]))
        .is_err());
}

#[test]
fn huge_empty_shape_does_not_overflow_before_zero_dimension() {
    let shape = vec![usize::MAX, usize::MAX, 0];
    let handle = crate::test_support::handle(
        17,
        shape.clone(),
        crate::NumericElementType::F64,
        crate::GpuTensorStorage::Real,
    );
    assert_eq!(handle.shape(), shape.as_slice());
    assert_eq!(handle.physical().element_capacity(), 0);
    assert_eq!(handle.physical().payload_bytes(), 0);
}

#[test]
fn native_logical_bytes_require_producer_evidence_not_public_view_flag() {
    let (logical, _) = crate::test_support::owned(
        17,
        vec![2],
        crate::NumericElementType::U8,
        crate::GpuTensorStorage::Real,
        true,
    );
    assert!(crate::handle_is_logical(&logical));
    assert_eq!(
        crate::handle_class_identity(&logical),
        Some(runmat_types::standard::LOGICAL.owned())
    );
    let numeric = crate::test_support::handle(
        17,
        vec![2],
        crate::NumericElementType::U8,
        crate::GpuTensorStorage::Real,
    );
    let mut wire = numeric.wire_descriptor();
    wire.view = wire.view.with_logical(true);
    assert!(numeric
        .allocation_lease()
        .downgrade()
        .acquire_tensor(&wire)
        .is_err());
    assert_eq!(
        crate::handle_class_identity(&numeric),
        Some(runmat_types::standard::UINT8.owned())
    );
}

#[test]
fn failed_explicit_retirement_retains_the_record_owner_without_live_reacquisition() {
    let owner = TestOwner::new(
        ID,
        AllocationRetirementOutcome::Uncertain {
            detail: "injected uncertain release".into(),
        },
    );
    let handle = tensor(&owner);
    let weak = owner.weak_lease(ID);
    let owner_weak = Arc::downgrade(&owner);
    drop(owner);
    let failure = handle.release().unwrap_err().physical.unwrap_err();
    assert_eq!(failure.identity(), ID);
    assert!(owner_weak.upgrade().is_some());
    assert!(weak.acquire().is_none());
    assert!(matches!(
        failure.outcome(),
        AllocationRetirementOutcome::Uncertain { .. }
    ));
    drop(failure);
    assert!(owner_weak.upgrade().is_none());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn release_and_weak_acquisition_race_never_retire_a_live_acquired_view() {
    for _ in 0..64 {
        let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
        let handle = tensor(&owner);
        let wire = handle.wire_descriptor();
        let weak = owner.weak_lease(ID);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let worker = {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                weak.acquire_tensor(&wire)
            })
        };
        barrier.wait();
        let outcome = handle.release().unwrap();
        if let Ok(acquired) = worker.join().unwrap() {
            assert_eq!(outcome, AllocationReleaseOutcome::ReferenceReleased);
            assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 0);
            assert_eq!(acquired.physical().element_capacity(), 2);
            drop(acquired);
        }
        assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    }
}
