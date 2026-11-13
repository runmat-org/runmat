use super::*;
use runmat_execution::resource::{
    RetainedHostAllocationClaim, RetainedHostViewError, RetainedHostViewGrant,
    RetainedHostViewIssuer,
};
use std::sync::{atomic::AtomicU64, Arc};

#[derive(Debug)]
struct TestViewIssuer {
    live: Arc<AtomicU64>,
    capacity: u64,
}

#[derive(Debug)]
struct TestViewGrant {
    bytes: u64,
    claim: Arc<TestViewClaim>,
    issuer: Arc<dyn RetainedHostViewIssuer>,
}

#[derive(Debug)]
struct TestViewClaim {
    bytes: u64,
    live: Arc<AtomicU64>,
}

impl Drop for TestViewClaim {
    fn drop(&mut self) {
        self.live.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

impl RetainedHostViewGrant for TestViewGrant {
    fn view_bytes(&self) -> u64 {
        self.bytes
    }

    fn issuer(&self) -> Arc<dyn RetainedHostViewIssuer> {
        self.issuer.clone()
    }

    fn backing_claim(&self) -> Arc<dyn RetainedHostAllocationClaim> {
        self.claim.clone()
    }
}

impl RetainedHostViewIssuer for TestViewIssuer {
    fn reserve_view(
        self: Arc<Self>,
        bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, RetainedHostViewError> {
        self.live
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= self.capacity)
            })
            .map_err(|_| RetainedHostViewError::ExceedsCapacity)?;
        Ok(Box::new(TestViewGrant {
            bytes,
            claim: Arc::new(TestViewClaim {
                bytes,
                live: self.live.clone(),
            }),
            issuer: self,
        }))
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn retained_view_grant_funds_one_record_while_clones_share_and_aliases_reserve_again() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let raw = tensor(&owner);
    let bytes = GpuTensorHandle::view_host_upper_bytes(2).unwrap();
    let live = Arc::new(AtomicU64::new(0));
    let issuer = Arc::new(TestViewIssuer {
        live: live.clone(),
        capacity: bytes * 2,
    });
    let grant = issuer.clone().reserve_view(bytes).unwrap();
    let published = raw
        .with_view_retained_grant(raw.view().clone(), grant)
        .unwrap();
    let same = published.clone();
    let weak = published.downgrade();
    assert_eq!(live.load(Ordering::Acquire), bytes);
    let alias =
        published
            .with_view(GpuTensorView::contiguous(vec![2, 1]).with_layout(
                GpuTensorLayout::Transpose(crate::TransposeInfo {
                    base_rows: 1,
                    base_cols: 2,
                }),
            ))
            .unwrap();
    assert_eq!(live.load(Ordering::Acquire), bytes * 2);
    assert!(matches!(
        published.with_provenance(crate::GpuHandleProvenance::Explicit),
        Err(AllocationAcquisitionError::RetainedHostView(
            RetainedHostViewError::ExceedsCapacity
        ))
    ));
    drop(raw);
    drop(published);
    assert_eq!(live.load(Ordering::Acquire), bytes * 2);
    drop(same);
    assert!(weak.acquire().is_err(), "dead view cannot be resurrected");
    assert_eq!(live.load(Ordering::Acquire), bytes * 2);
    drop(alias);
    assert_eq!(live.load(Ordering::Acquire), bytes);
    drop(weak);
    assert_eq!(live.load(Ordering::Acquire), 0);

    // A finite owner can reacquire a wire only by funding a fresh view from
    // the authenticated finite lease. Publication's supplied grant is used
    // once; acquisition and the direct constructor cannot evade the issuer.
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let live = Arc::new(AtomicU64::new(0));
    let issuer = Arc::new(TestViewIssuer {
        live: live.clone(),
        capacity: bytes * 2,
    });
    let lease = AllocationLease::register_with_retained_metadata(
        ID,
        owner.clone(),
        Some(RetainedLeaseHostMetadata::new(Arc::new(()), issuer.clone())),
    )
    .unwrap();
    let initial = GpuTensorHandle::from_allocation_with_retained_grant(
        lease,
        GpuTensorView::contiguous(vec![1, 2]),
        issuer.clone().reserve_view(bytes).unwrap(),
    )
    .unwrap();
    assert_eq!(live.load(Ordering::Acquire), bytes);
    let foreign_live = Arc::new(AtomicU64::new(0));
    let foreign_issuer = Arc::new(TestViewIssuer {
        live: foreign_live.clone(),
        capacity: bytes,
    });
    assert_eq!(
        GpuTensorHandle::from_allocation_with_retained_grant(
            initial.allocation_lease().clone(),
            initial.view().clone(),
            foreign_issuer.reserve_view(bytes).unwrap(),
        )
        .unwrap_err(),
        AllocationAcquisitionError::WrongOwner
    );
    assert_eq!(foreign_live.load(Ordering::Acquire), 0);
    let wire = initial.wire_descriptor();
    let weak_lease = owner.weak_lease(ID);
    let reacquired = weak_lease.acquire_tensor(&wire).unwrap();
    assert_eq!(live.load(Ordering::Acquire), bytes * 2);
    assert!(matches!(
        weak_lease.acquire_tensor(&wire),
        Err(AllocationAcquisitionError::RetainedHostView(
            RetainedHostViewError::ExceedsCapacity
        ))
    ));
    drop(reacquired);
    let direct = GpuTensorHandle::from_allocation(
        initial.allocation_lease().clone(),
        initial.view().clone(),
    )
    .unwrap();
    assert_eq!(live.load(Ordering::Acquire), bytes * 2);
    drop(direct);
    drop(initial);
    assert_eq!(live.load(Ordering::Acquire), 0);
}

fn tensor(owner: &Shared<TestOwner>) -> GpuTensorHandle {
    GpuTensorHandle::from_allocation(
        AllocationLease::register(ID, owner.clone()).unwrap(),
        GpuTensorView::contiguous(vec![1, 2]),
    )
    .unwrap()
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unchanged_provenance_retains_view_identity_but_changed_intent_mints_a_view() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let original = tensor(&owner);
    let explicit = original
        .with_provenance(crate::GpuHandleProvenance::Explicit)
        .unwrap();
    assert_ne!(explicit.view_identity(), original.view_identity());
    assert_eq!(original.view().provenance(), None);
    let published_again = explicit
        .with_provenance(crate::GpuHandleProvenance::Explicit)
        .unwrap();
    assert_eq!(published_again.view_identity(), explicit.view_identity());
    assert_eq!(published_again.shape().as_ptr(), explicit.shape().as_ptr());
    let automatic = explicit
        .with_provenance(crate::GpuHandleProvenance::Automatic)
        .unwrap();
    assert_ne!(automatic.view_identity(), explicit.view_identity());
    assert_eq!(
        original.release().unwrap(),
        AllocationReleaseOutcome::ReferenceReleased
    );
    assert_eq!(
        explicit.release().unwrap(),
        AllocationReleaseOutcome::ReferenceReleased
    );
    assert_eq!(
        published_again.release().unwrap(),
        AllocationReleaseOutcome::ReferenceReleased
    );
    assert_eq!(
        automatic.release().unwrap(),
        AllocationReleaseOutcome::Retired
    );
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn unchanged_provenance_does_not_bypass_a_retiring_owner() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let original = tensor(&owner)
        .with_provenance(crate::GpuHandleProvenance::Explicit)
        .unwrap();
    let live = owner.weak_lease(ID);
    owner.entries.lock().unwrap().get_mut(&ID).unwrap().phase = Entry::Retiring;
    assert_eq!(
        original
            .with_provenance(crate::GpuHandleProvenance::Explicit)
            .unwrap_err(),
        AllocationAcquisitionError::NotLive
    );
    owner.entries.lock().unwrap().get_mut(&ID).unwrap().phase = Entry::Live(live);
    drop(original);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
    assert_eq!(handle.view().checked_element_count(), Ok(0));
    assert!(handle
        .view()
        .clone()
        .with_shape(vec![usize::MAX, usize::MAX])
        .checked_element_count()
        .is_err());
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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn failed_explicit_retirement_retains_the_record_owner_without_live_reacquisition() {
    let owner = TestOwner::new(
        ID,
        AllocationRetirementOutcome::Uncertain {
            detail: "injected uncertain release".into(),
        },
    );
    let handle = tensor(&owner);
    let weak = owner.weak_lease(ID);
    let owner_weak = Shared::downgrade(&owner);
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
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
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
