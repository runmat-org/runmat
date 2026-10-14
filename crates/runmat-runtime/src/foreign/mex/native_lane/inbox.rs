use super::BoundaryEnvelope;
use futures::channel::mpsc;
use futures::Stream;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

/// A single receiver may have foreground and background waiters. Each poll
/// borrows it only for `poll_next`; the shared channel wake fans out to every
/// live waiter without retaining a cancelled future's waker.
pub(super) struct CallbackInbox {
    receiver: RefCell<mpsc::UnboundedReceiver<BoundaryEnvelope>>,
    wake: Arc<InboxWake>,
    next_waiter: Cell<u64>,
}

struct InboxWake {
    waiters: Mutex<HashMap<u64, Waker>>,
}

impl Wake for InboxWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let waiters = self
            .waiters
            .lock()
            .expect("MEX inbox wake table is not poisoned");
        let current = waiters.values().cloned().collect::<Vec<_>>();
        drop(waiters);
        for waiter in current {
            waiter.wake();
        }
    }
}

impl CallbackInbox {
    pub(super) fn new(receiver: mpsc::UnboundedReceiver<BoundaryEnvelope>) -> Self {
        Self {
            receiver: RefCell::new(receiver),
            wake: Arc::new(InboxWake {
                waiters: Mutex::new(HashMap::new()),
            }),
            next_waiter: Cell::new(1),
        }
    }

    pub(super) fn next(&self) -> ReceiveTicket<'_> {
        let id = self.next_waiter.get();
        self.next_waiter.set(
            id.checked_add(1)
                .expect("MEX inbox waiter identity exhausted"),
        );
        ReceiveTicket { inbox: self, id }
    }

    pub(super) fn close_and_drain(&self) -> Vec<BoundaryEnvelope> {
        let mut receiver = self.receiver.borrow_mut();
        receiver.close();
        let mut drained = Vec::new();
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        while let Poll::Ready(Some(envelope)) = Pin::new(&mut *receiver).poll_next(&mut cx) {
            drained.push(envelope);
        }
        drained
    }

    pub(super) fn pop_ready(&self) -> Option<BoundaryEnvelope> {
        let mut receiver = self.receiver.borrow_mut();
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        match Pin::new(&mut *receiver).poll_next(&mut cx) {
            Poll::Ready(Some(envelope)) => Some(envelope),
            _ => None,
        }
    }
}

pub(super) struct ReceiveTicket<'a> {
    inbox: &'a CallbackInbox,
    id: u64,
}

impl Future for ReceiveTicket<'_> {
    type Output = Option<BoundaryEnvelope>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut waiters = self
            .inbox
            .wake
            .waiters
            .lock()
            .expect("MEX inbox wake table is not poisoned");
        waiters.insert(self.id, cx.waker().clone());
        drop(waiters);
        let wake = Waker::from(Arc::clone(&self.inbox.wake));
        let mut receiver = self.inbox.receiver.borrow_mut();
        Pin::new(&mut *receiver).poll_next(&mut Context::from_waker(&wake))
    }
}

impl Drop for ReceiveTicket<'_> {
    fn drop(&mut self) {
        let removed = self
            .inbox
            .wake
            .waiters
            .lock()
            .expect("MEX inbox wake table is not poisoned")
            .remove(&self.id);
        drop(removed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountWake(AtomicUsize);

    impl Wake for CountWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn queued_envelope_wakes_both_live_receive_tickets() {
        let (sender, receiver) = mpsc::unbounded();
        let inbox = CallbackInbox::new(receiver);
        let mut first = Box::pin(inbox.next());
        let mut second = Box::pin(inbox.next());
        let first_count = Arc::new(CountWake(AtomicUsize::new(0)));
        let second_count = Arc::new(CountWake(AtomicUsize::new(0)));
        let first_waker = Waker::from(Arc::clone(&first_count));
        let second_waker = Waker::from(Arc::clone(&second_count));
        assert!(matches!(
            first.as_mut().poll(&mut Context::from_waker(&first_waker)),
            Poll::Pending
        ));
        assert!(matches!(
            second
                .as_mut()
                .poll(&mut Context::from_waker(&second_waker)),
            Poll::Pending
        ));
        sender
            .unbounded_send(BoundaryEnvelope::ServiceFinished { service_id: 1 })
            .expect("publish envelope");
        assert!(first_count.0.load(Ordering::SeqCst) > 0);
        assert!(second_count.0.load(Ordering::SeqCst) > 0);
    }
}
