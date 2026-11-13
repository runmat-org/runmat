use super::*;
use crate::observation::{
    ProviderObservationFailure, ProviderObservationStatus, ProviderOperationReceipt,
    ProviderResourcePhasePlan, ProviderResourceTicket,
};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use std::task::{Wake, Waker};

struct Noop;
impl Wake for Noop {
    fn wake(self: Arc<Self>) {}
}

fn scope(receipt: &ProviderOperationReceipt) -> ProviderResourceScope {
    let scope = receipt
        .recorder()
        .begin_resource_scope(ProviderResourcePhasePlan::new([64]).unwrap())
        .unwrap();
    scope.begin_next_phase().unwrap();
    scope
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn interleaved_pending_ready_and_nested_polls_restore_ambient_scope() {
    let mut receipt = ProviderOperationReceipt::new();
    let outer = scope(&receipt);
    let inner = scope(&receipt);
    let outer_id = outer.id();
    let inner_id = inner.id();
    let inner_for_poll = inner.clone();
    let mut polls = 0;
    let mut future = Box::pin(observe_provider_resource_future(
        Some(outer.clone()),
        std::future::poll_fn(move |cx| {
            assert_eq!(current_provider_resource_scope().unwrap().id(), outer_id);
            let mut nested = Box::pin(observe_provider_resource_future(
                Some(inner_for_poll.clone()),
                async move {
                    assert_eq!(current_provider_resource_scope().unwrap().id(), inner_id);
                },
            ));
            assert!(nested.as_mut().poll(cx).is_ready());
            assert_eq!(current_provider_resource_scope().unwrap().id(), outer_id);
            polls += 1;
            if polls == 1 {
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        }),
    ));
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    assert!(current_provider_resource_scope().is_none());
    assert!(future.as_mut().poll(&mut cx).is_pending());
    assert!(current_provider_resource_scope().is_none());
    assert!(future.as_mut().poll(&mut cx).is_ready());
    assert!(current_provider_resource_scope().is_none());
    drop(future);
    inner.finish().unwrap();
    outer.finish().unwrap();
    receipt.publish().unwrap();
    assert!(matches!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Completed { .. }
    ));
}

struct DropObserved {
    expected_id: u64,
    seen: Rc<Cell<bool>>,
}

impl Future for DropObserved {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

impl Drop for DropObserved {
    fn drop(&mut self) {
        assert_eq!(
            current_provider_resource_scope().unwrap().id(),
            self.expected_id
        );
        self.seen.set(true);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unpolled_inner_drop_uses_issued_scope_then_restores_outer() {
    let mut receipt = ProviderOperationReceipt::new();
    let outer = scope(&receipt);
    let inner = scope(&receipt);
    let seen = Rc::new(Cell::new(false));
    let future = observe_provider_resource_future(
        Some(inner.clone()),
        DropObserved {
            expected_id: inner.id(),
            seen: seen.clone(),
        },
    );
    with_scope(Some(outer.clone()), || {
        assert_eq!(current_provider_resource_scope().unwrap().id(), outer.id());
        drop(future);
        assert_eq!(current_provider_resource_scope().unwrap().id(), outer.id());
    });
    assert!(seen.get());
    assert!(current_provider_resource_scope().is_none());
    inner.finish().unwrap();
    outer.finish().unwrap();
    receipt.publish().unwrap();
    assert!(matches!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Completed { .. }
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn issued_ticket_retires_outside_ambient_after_canonical_cancellation() {
    let mut receipt = ProviderOperationReceipt::new();
    let issued = scope(&receipt);
    let holder: Rc<RefCell<Option<ProviderResourceTicket>>> = Rc::new(RefCell::new(None));
    let holder_for_poll = holder.clone();
    let expected_id = issued.id();
    let mut future = Box::pin(observe_provider_resource_future(
        Some(issued.clone()),
        async move {
            let active = current_provider_resource_scope().unwrap();
            assert_eq!(active.id(), expected_id);
            *holder_for_poll.borrow_mut() = Some(active.issue_resource_ticket(32).unwrap());
        },
    ));
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    assert!(future.as_mut().poll(&mut cx).is_ready());
    assert!(current_provider_resource_scope().is_none());
    issued.finish().unwrap();
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    receipt.fail(ProviderObservationFailure::Cancelled);
    drop(future);
    drop(issued);
    drop(holder.borrow_mut().take());
    let evidence = receipt.take_completed().unwrap();
    assert_eq!(
        evidence.status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::Cancelled)
    );
    assert_eq!(evidence.resources.unwrap().requested_live_bytes, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn ready_error_restores_ambient_and_owner_failure_remains_distinct() {
    let mut receipt = ProviderOperationReceipt::new();
    let issued = scope(&receipt);
    let mut future = Box::pin(observe_provider_resource_future(
        Some(issued.clone()),
        async {
            assert!(current_provider_resource_scope().is_some());
            Err::<(), _>("owner error")
        },
    ));
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    assert!(matches!(
        future.as_mut().poll(&mut cx),
        Poll::Ready(Err("owner error"))
    ));
    assert!(current_provider_resource_scope().is_none());
    issued.fail(ProviderObservationFailure::ProviderFailed);
    drop(future);
    drop(issued);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed)
    );
}

#[cfg(all(not(target_arch = "wasm32"), panic = "unwind"))]
#[test]
fn panicking_poll_restores_ambient_and_does_not_upgrade_receipt() {
    let mut receipt = ProviderOperationReceipt::new();
    let issued = scope(&receipt);
    let mut future = Box::pin(observe_provider_resource_future(
        Some(issued.clone()),
        std::future::poll_fn(|_| -> Poll<()> {
            assert!(current_provider_resource_scope().is_some());
            panic!("injected poll unwind");
        }),
    ));
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = future.as_mut().poll(&mut cx);
    }));
    assert!(panicked.is_err());
    assert!(current_provider_resource_scope().is_none());
    issued.fail(ProviderObservationFailure::ProviderFailed);
    drop(future);
    drop(issued);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed)
    );
}
