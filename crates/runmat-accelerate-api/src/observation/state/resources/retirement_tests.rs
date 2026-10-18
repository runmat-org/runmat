use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Weak,
};
use std::task::{Context, Poll, Wake, Waker};

#[derive(Default)]
struct WakeCount(AtomicUsize);
impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct ReentrantWake {
    owner: Weak<Shared>,
    wakes: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl ReentrantWake {
    fn assert_unlocked(&self) {
        if let Some(owner) = self.owner.upgrade() {
            assert!(owner.state.try_lock().is_ok(), "waker ran under state lock");
        }
    }
}

impl Wake for ReentrantWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.assert_unlocked();
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for ReentrantWake {
    fn drop(&mut self) {
        self.assert_unlocked();
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn reentrant_waker(
    receipt: &ProviderOperationReceipt,
    wakes: &Arc<AtomicUsize>,
    drops: &Arc<AtomicUsize>,
) -> Waker {
    Arc::new(ReentrantWake {
        owner: Arc::downgrade(&receipt.0),
        wakes: wakes.clone(),
        drops: drops.clone(),
    })
    .into()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn no_scope_is_distinct_from_a_scope_with_all_tickets_retired() {
    let mut receipt = ProviderOperationReceipt::new();
    let wake: Waker = Arc::new(WakeCount::default()).into();
    let mut context = Context::from_waker(&wake);
    assert_eq!(
        receipt.poll_resource_retirement(&mut context),
        Poll::Ready(None)
    );
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([8]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    let ticket = scope.issue_resource_ticket(8).unwrap();
    assert_eq!(
        receipt.poll_resource_retirement(&mut context),
        Poll::Pending
    );
    drop(ticket);
    let Poll::Ready(Some(snapshot)) = receipt.poll_resource_retirement(&mut context) else {
        panic!("last actual owner retired");
    };
    assert_eq!((snapshot.issued, snapshot.retired), (1, 1));
    assert_eq!(snapshot.requested_live_bytes, 0);
    // Retirement does not finish the scope or certify successful execution.
    assert!(!snapshot.scopes[0].finished);
    receipt.fail(ProviderObservationFailure::Cancelled);
    drop(scope);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn failure_feedback_may_be_taken_before_late_final_owner_retirement() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([16]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    let ticket = scope.issue_resource_ticket(16).unwrap();
    let alias = ticket.clone();
    receipt.fail(ProviderObservationFailure::Cancelled);
    drop(scope);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    let counter = Arc::new(WakeCount::default());
    let wake: Waker = counter.clone().into();
    let mut context = Context::from_waker(&wake);
    assert_eq!(
        receipt.poll_resource_retirement(&mut context),
        Poll::Pending
    );
    assert_eq!(
        receipt.poll_resource_quiescence(&mut context),
        Poll::Pending
    );
    drop(ticket);
    assert_eq!(counter.0.load(Ordering::SeqCst), 0);
    assert_eq!(
        receipt.poll_resource_retirement(&mut context),
        Poll::Pending
    );
    drop(alias);
    assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    let Poll::Ready(Some(snapshot)) = receipt.poll_resource_retirement(&mut context) else {
        panic!("last shared owner retired");
    };
    assert_eq!((snapshot.issued, snapshot.retired), (1, 1));
    assert_eq!(snapshot.requested_live_bytes, 0);
    assert!(!snapshot.scopes[0].finished);
    assert_eq!(
        receipt.poll_resource_quiescence(&mut context),
        Poll::Ready(Err(ProviderObservationFailure::Cancelled))
    );
    assert_eq!(
        receipt.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn operation_metadata_survives_dropped_waiter_until_final_ticket() {
    let mut receipt = ProviderOperationReceipt::new();
    let funding = Arc::new(AtomicUsize::new(0));
    let retained = Arc::downgrade(&funding);
    receipt.retain_operation_host_metadata(funding).unwrap();
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([8]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    let ticket = scope.issue_resource_ticket(8).unwrap();
    drop(scope);
    drop(receipt);
    assert!(
        retained.upgrade().is_some(),
        "callback owner retains funding"
    );
    drop(ticket);
    assert!(retained.upgrade().is_none(), "last owner releases funding");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn last_ticket_retired_before_first_poll_is_ready_without_a_waker() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([8]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    let ticket = scope.issue_resource_ticket(8).unwrap();
    receipt.fail(ProviderObservationFailure::Cancelled);
    drop(scope);
    drop(ticket);
    let wake: Waker = Arc::new(WakeCount::default()).into();
    let mut context = Context::from_waker(&wake);
    let Poll::Ready(Some(snapshot)) = receipt.poll_resource_retirement(&mut context) else {
        panic!("already retired owner must not wait");
    };
    assert_eq!((snapshot.issued, snapshot.retired), (1, 1));
    assert_eq!(snapshot.requested_live_bytes, 0);
    assert_eq!(
        receipt.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn replaced_waiter_drop_and_both_wakes_reenter_after_unlock() {
    let mut receipt = ProviderOperationReceipt::new();
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([8]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    let ticket = scope.issue_resource_ticket(8).unwrap();
    let operation_wakes = Arc::new(AtomicUsize::new(0));
    let operation_drops = Arc::new(AtomicUsize::new(0));
    let old_wakes = Arc::new(AtomicUsize::new(0));
    let old_drops = Arc::new(AtomicUsize::new(0));
    let resource_wakes = Arc::new(AtomicUsize::new(0));
    let resource_drops = Arc::new(AtomicUsize::new(0));

    {
        let waker = reentrant_waker(&receipt, &operation_wakes, &operation_drops);
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            receipt.poll_completed(&mut context),
            Poll::Pending
        ));
    }
    {
        let waker = reentrant_waker(&receipt, &old_wakes, &old_drops);
        let mut context = Context::from_waker(&waker);
        assert_eq!(
            receipt.poll_resource_retirement(&mut context),
            Poll::Pending
        );
    }
    {
        let waker = reentrant_waker(&receipt, &resource_wakes, &resource_drops);
        let mut context = Context::from_waker(&waker);
        assert_eq!(
            receipt.poll_resource_retirement(&mut context),
            Poll::Pending
        );
    }
    assert_eq!(old_drops.load(Ordering::SeqCst), 1);
    assert_eq!(old_wakes.load(Ordering::SeqCst), 0);
    receipt.fail(ProviderObservationFailure::Cancelled);
    assert_eq!(operation_wakes.load(Ordering::SeqCst), 1);
    assert_eq!(resource_wakes.load(Ordering::SeqCst), 0);
    drop(scope);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    drop(ticket);
    assert_eq!(resource_wakes.load(Ordering::SeqCst), 1);
    assert_eq!(operation_drops.load(Ordering::SeqCst), 1);
    assert_eq!(resource_drops.load(Ordering::SeqCst), 1);
    let wake: Waker = Arc::new(WakeCount::default()).into();
    let mut context = Context::from_waker(&wake);
    assert!(matches!(
        receipt.poll_resource_retirement(&mut context),
        Poll::Ready(Some(_))
    ));
}
