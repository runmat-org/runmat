use super::owner::{Entry, RegistrationFault, TestOwner};
use super::*;

#[test]
fn weak_publication_is_unavailable_inside_registration_and_available_after_ok() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    assert!(owner
        .observed_unpublished_registration
        .load(Ordering::SeqCst));
    let acquired = owner.weak_lease(ID).acquire().expect("published after Ok");
    assert!(acquired.shares_allocation_with(&lease));
    drop(acquired);
    drop(lease);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
}

fn payload_pointer(owner: &TestOwner) -> *const u8 {
    let entries = owner
        .entries
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let record = entries.get(&ID).unwrap();
    let payload = record.payload.as_ref().expect("payload remains owned");
    assert_eq!(&**payload, &[19, 23]);
    payload.as_ptr()
}

fn retry_and_check(owner: &Shared<TestOwner>, pointer: *const u8) {
    assert_eq!(
        payload_pointer(owner),
        pointer,
        "no payload replacement or orphaning"
    );
    {
        let entries = owner
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        assert!(matches!(
            entries.get(&ID).unwrap().phase,
            Entry::Unregistered
        ));
    }
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 0);
    *owner.registration_fault.lock().unwrap() = RegistrationFault::None;
    let lease =
        AllocationLease::register(ID, owner.clone()).expect("registration slot remains reusable");
    assert_eq!(payload_pointer(owner), pointer);
    drop(lease);
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    assert_eq!(owner.outcome(ID), AllocationRetirementOutcome::Retired);
}

#[test]
fn rejected_admission_preserves_payload_and_reusable_registration_slot() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let pointer = payload_pointer(&owner);
    *owner.registration_fault.lock().unwrap() = RegistrationFault::RejectBeforeAssociation;
    assert_eq!(
        AllocationLease::register(ID, owner.clone()).unwrap_err(),
        AllocationRegistrationError::NotLive
    );
    retry_and_check(&owner, pointer);
}

#[cfg(panic = "unwind")]
#[test]
fn pre_association_unwind_preserves_payload_and_reusable_registration_slot() {
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    let pointer = payload_pointer(&owner);
    *owner.registration_fault.lock().unwrap() = RegistrationFault::PanicBeforeAssociation;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        AllocationLease::register(ID, owner.clone())
    }));
    assert!(result.is_err());
    retry_and_check(&owner, pointer);
}

#[cfg(panic = "unwind")]
#[test]
fn retirement_panic_payload_destructor_is_contained_without_losing_failure_record() {
    struct Payload;
    impl Drop for Payload {
        fn drop(&mut self) {
            panic!("panic payload destructor failed");
        }
    }
    let owner = TestOwner::new(ID, AllocationRetirementOutcome::Retired);
    owner.set_callback(|| std::panic::panic_any(Payload));
    let lease = AllocationLease::register(ID, owner.clone()).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(lease)));
    assert!(result.is_ok());
    assert_eq!(owner.retire_calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        owner.outcome(ID),
        AllocationRetirementOutcome::Uncertain { .. }
    ));
    assert_eq!(
        &**owner
            .entries
            .lock()
            .unwrap()
            .get(&ID)
            .unwrap()
            .payload
            .as_ref()
            .unwrap(),
        &[19, 23]
    );
}
