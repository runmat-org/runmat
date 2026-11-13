mod owner;
mod publication;
mod tensor;
mod native_export;

use super::shared::Shared;
use super::*;
use owner::{Entry, TestOwner};
use runmat_execution::resource::{
    RetainedHostViewError, RetainedHostViewGrant, RetainedHostViewIssuer,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[cfg(all(target_arch = "wasm32", not(feature = "wgpu")))]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

const ID: AllocationIdentity = AllocationIdentity::new(17, 41, 2);

#[derive(Debug)]
struct TestLeaseClaim(Arc<AtomicUsize>);

#[derive(Debug)]
struct TestLeaseIssuer;

impl RetainedHostViewIssuer for TestLeaseIssuer {
    fn reserve_view(
        self: Arc<Self>,
        _view_bytes: u64,
    ) -> Result<Box<dyn RetainedHostViewGrant>, RetainedHostViewError> {
        Err(RetainedHostViewError::Revoked)
    }
}

impl Drop for TestLeaseClaim {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn clones_retain_captured_owner_and_retire_exactly_once() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let owner_weak = Shared::downgrade(&owner);
    let claimed = Arc::new(AtomicUsize::new(1));
    let claim = Arc::new(TestLeaseClaim(claimed.clone()));
    let lease = AllocationLease::register_with_retained_metadata(
        ID,
        owner.clone(),
        Some(RetainedLeaseHostMetadata::new(
            claim,
            Arc::new(TestLeaseIssuer),
        )),
    )
    .unwrap();
    let second = lease.clone();
    let weak = lease.downgrade();
    let third = weak.acquire().unwrap();
    let map_alias = owner.weak_lease(ID).acquire().unwrap();
    assert!(map_alias.shares_allocation_with(&lease));
    drop(map_alias);
    assert_eq!(lease.identity(), ID);
    assert_eq!(
        (ID.device_id(), ID.allocation_id(), ID.generation()),
        (17, 41, 2)
    );
    assert!(lease.shares_allocation_with(&second));
    assert!(lease.shares_allocation_with(&third));
    drop(lease);
    drop(second);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 0);
    drop(owner);
    assert!(
        owner_weak.upgrade().is_some(),
        "lease captures owner lifetime"
    );
    drop(third);
    assert!(weak.acquire().is_none());
    assert_eq!(claimed.load(Ordering::SeqCst), 1);
    assert!(
        owner_weak.upgrade().is_none(),
        "weak map association creates no state cycle"
    );
    drop(weak);
    assert_eq!(claimed.load(Ordering::SeqCst), 0);
}

#[test]
fn rejected_registration_never_retires_an_existing_allocation() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    assert_eq!(
        AllocationLease::register(ID, owner.clone()).unwrap_err(),
        AllocationRegistrationError::AlreadyRegistered
    );
    for (id, expected) in [
        (
            AllocationIdentity::new(18, 41, 2),
            AllocationRegistrationError::WrongOwner,
        ),
        (
            AllocationIdentity::new(17, 42, 2),
            AllocationRegistrationError::UnknownAllocation,
        ),
        (
            AllocationIdentity::new(17, 41, 3),
            AllocationRegistrationError::StaleGeneration,
        ),
    ] {
        assert_eq!(
            AllocationLease::register(id, owner.clone()).unwrap_err(),
            expected
        );
    }
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 0);
    drop(lease);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        AllocationLease::register(ID, owner.clone()).unwrap_err(),
        AllocationRegistrationError::NotLive
    );
}

#[test]
fn retirement_outcomes_remain_in_the_existing_owner_without_automatic_retry() {
    for outcome in [
        AllocationRetirementOutcome::Retired,
        AllocationRetirementOutcome::RetainedFailure {
            detail: "payload retained without retry admission".into(),
        },
        AllocationRetirementOutcome::RetryableFailure {
            detail: "no release occurred".into(),
        },
        AllocationRetirementOutcome::Uncertain {
            detail: "driver result inconclusive".into(),
        },
        AllocationRetirementOutcome::DeviceLost {
            detail: "context lost".into(),
        },
    ] {
        let owner = TestOwner::new(ID, outcome.clone());
        let lease = AllocationLease::register(ID, owner.clone()).unwrap();
        let weak = lease.downgrade();
        drop(lease);
        assert!(weak.acquire().is_none());
        assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
        assert_eq!(owner.outcome(ID), outcome);
        assert_eq!(
            AllocationLease::register(ID, owner.clone()).unwrap_err(),
            AllocationRegistrationError::NotLive
        );
        assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_upload_failure_keeps_the_last_custom_owner_until_error_drop() {
    let owner = TestOwner::new(
        ID,
        AllocationRetirementOutcome::RetainedFailure {
            detail: "bookkeeping rejected retirement".into(),
        },
    );
    let weak_owner = Shared::downgrade(&owner);
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    let handle =
        GpuTensorHandle::from_allocation(lease, GpuTensorView::contiguous(vec![1])).unwrap();
    drop(owner);
    let error = crate::InvalidNumericUploadResult::reject(handle, "malformed output".into());
    assert!(crate::is_invalid_numeric_upload_result(&error));
    let crate::NumericUploadError::Invalid(invalid) = &error else {
        panic!("invalid result was erased")
    };
    let failed = invalid
        .cleanup_error()
        .unwrap()
        .physical
        .as_ref()
        .unwrap_err();
    assert_eq!(failed.identity(), ID);
    assert!(matches!(
        failed.outcome(),
        AllocationRetirementOutcome::RetainedFailure { .. }
    ));
    assert!(weak_owner.upgrade().is_some());
    drop(error);
    assert!(
        weak_owner.upgrade().is_none(),
        "caught/dropped error releases its owner without a cycle"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_upload_error_preserves_thread_safe_error_transport() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<crate::NumericUploadError>();
    let error: anyhow::Error = crate::NumericUploadError::from(anyhow::anyhow!("transport")).into();
    assert_eq!(error.to_string(), "transport");
}

#[test]
fn retirement_callback_can_reenter_owner_and_cannot_resurrect_lease() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    let weak = lease.downgrade();
    let owner_weak = Shared::downgrade(&owner);
    owner.set_callback(move || {
        let owner = owner_weak.upgrade().unwrap();
        let entries = owner
            .entries
            .try_lock()
            .expect("no map lock crosses retirement callback");
        assert!(matches!(
            entries.get(&ID).map(|record| &record.phase),
            Some(Entry::Retiring)
        ));
        assert!(weak.acquire().is_none());
    });
    drop(lease);
    assert_eq!(owner.outcome(ID), AllocationRetirementOutcome::Retired);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn an_acquired_reference_wins_before_final_drop() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    let weak = lease.downgrade();
    let acquired = weak.acquire().unwrap();
    drop(lease);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 0);
    drop(acquired);
    assert!(weak.acquire().is_none());
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn final_drop_wins_before_a_racing_weak_acquisition() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AllocationLease>();
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    let weak = lease.downgrade();
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (resume_tx, resume_rx) = std::sync::mpsc::channel();
    owner.set_callback(move || {
        entered_tx.send(()).unwrap();
        resume_rx.recv().unwrap();
    });
    let dropping = std::thread::spawn(move || drop(lease));
    entered_rx.recv().unwrap();
    let acquired = weak.acquire();
    resume_tx.send(()).unwrap();
    dropping.join().unwrap();
    assert!(
        acquired.is_none(),
        "final drop cannot be resurrected during backend work"
    );
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
}

#[cfg(panic = "unwind")]
#[test]
fn unexpected_retirement_unwind_is_recorded_uncertain_and_not_retried() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    owner.panic_during_retire.store(true, Ordering::SeqCst);
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(lease)));
    assert!(
        result.is_ok(),
        "lease Drop must not propagate an owner unwind"
    );
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        owner.outcome(ID),
        AllocationRetirementOutcome::Uncertain { .. }
    ));
}

#[cfg(panic = "unwind")]
#[test]
fn retirement_failure_during_outer_unwind_preserves_the_original_failure() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    owner.panic_during_retire.store(true, Ordering::SeqCst);
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _lease = lease;
        panic!("original invocation failure");
    }));
    let panic = result.expect_err("original failure is preserved");
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"original invocation failure")
    );
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        owner.outcome(ID),
        AllocationRetirementOutcome::Uncertain { .. }
    ));
}
