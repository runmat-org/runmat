use std::cell::RefCell;
use std::rc::Rc;

use super::super::IsolatedMexClient;

enum SlotState<T> {
    Available(T),
    InFlight,
}

/// The session's transport owner. A command takes the client out of the
/// short-lived RefCell borrow before polling any callback or process I/O.
pub(super) struct OperationSlot<T> {
    state: RefCell<SlotState<T>>,
}

pub(super) type IsolatedClientSlot = OperationSlot<IsolatedMexClient>;

impl<T> OperationSlot<T> {
    pub(super) fn new(client: T) -> Rc<Self> {
        Rc::new(Self {
            state: RefCell::new(SlotState::Available(client)),
        })
    }

    pub(super) fn acquire(self: &Rc<Self>) -> Result<OperationLease<T>, TransportBusy> {
        let client = {
            let mut state = self.state.borrow_mut();
            match std::mem::replace(&mut *state, SlotState::InFlight) {
                SlotState::Available(client) => client,
                SlotState::InFlight => return Err(TransportBusy),
            }
        };
        Ok(OperationLease {
            slot: Rc::clone(self),
            client: Some(client),
        })
    }
}

#[derive(Debug)]
pub(super) struct TransportBusy;

/// Cancellation drops the future's command state before this lease. Restore
/// the still-live client (including its pending protocol record) before the
/// session's path-activity guard permits another operation.
pub(super) struct OperationLease<T> {
    slot: Rc<OperationSlot<T>>,
    client: Option<T>,
}

impl<T> OperationLease<T> {
    pub(super) fn client(&mut self) -> &mut T {
        self.client
            .as_mut()
            .expect("isolated client operation owns its transport")
    }
}

impl<T> Drop for OperationLease<T> {
    fn drop(&mut self) {
        let mut state = self.slot.state.borrow_mut();
        assert!(
            matches!(*state, SlotState::InFlight),
            "isolated MEX transport acquired twice"
        );
        *state = SlotState::Available(
            self.client
                .take()
                .expect("isolated client operation owns its transport"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use std::cell::Cell;

    struct TransportProbe(Rc<Cell<usize>>);

    impl Drop for TransportProbe {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn busy_reentry_and_cancel_restore_the_same_transport() {
        let drops = Rc::new(Cell::new(0));
        let slot = OperationSlot::new(TransportProbe(Rc::clone(&drops)));
        let mut operation = slot.acquire().expect("first admission");
        assert!(slot.acquire().is_err(), "reentry is busy, not absent");
        assert!(Rc::ptr_eq(&operation.client().0, &drops));
        let suspended = async move {
            let _operation = operation;
            futures::future::pending::<()>().await;
        };
        assert!(suspended.now_or_never().is_none()); // cancel while suspended
        assert_eq!(drops.get(), 0);
        let operation = slot.acquire().expect("restored admission");
        drop(operation);
        drop(slot);
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn removed_session_entry_cannot_republish_an_inflight_transport() {
        let drops = Rc::new(Cell::new(0));
        let session_entry = OperationSlot::new(TransportProbe(Rc::clone(&drops)));
        let operation = session_entry.acquire().expect("first admission");
        drop(session_entry);
        assert_eq!(
            drops.get(),
            0,
            "in-flight operation still owns the transport"
        );
        drop(operation);
        assert_eq!(drops.get(), 1, "orphan slot and transport retire together");
    }
}
