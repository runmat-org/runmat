//! Atomic host acknowledgement for a callback admitted to the session lane.

use futures::channel::oneshot;
use runmat_value::{
    GraphicsCallbackDisposition, GraphicsCallbackHostFailure, GraphicsCallbackResult,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

const QUEUED: u8 = 0;
const RETIRED: u8 = 1;
const DISPATCHED: u8 = 2;
const COMMITTED: u8 = 3;

pub(super) struct HostReply {
    state: AtomicU8,
    sender:
        Mutex<Option<oneshot::Sender<Result<GraphicsCallbackResult, GraphicsCallbackHostFailure>>>>,
}

impl HostReply {
    pub(super) fn new() -> (Arc<Self>, GraphicsCallbackHostTicket) {
        let (sender, receiver) = oneshot::channel();
        let reply = Arc::new(Self {
            state: AtomicU8::new(QUEUED),
            sender: Mutex::new(Some(sender)),
        });
        let ticket = GraphicsCallbackHostTicket {
            reply: Arc::clone(&reply),
            receiver,
        };
        (reply, ticket)
    }

    pub(super) fn is_retired(&self) -> bool {
        self.state.load(Ordering::Acquire) == RETIRED
    }

    pub(super) fn begin(&self) -> bool {
        self.state
            .compare_exchange(QUEUED, DISPATCHED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub(super) fn disposition(&self) -> GraphicsCallbackDisposition {
        match self.state.load(Ordering::Acquire) {
            QUEUED | RETIRED => GraphicsCallbackDisposition::NotPublished,
            DISPATCHED => GraphicsCallbackDisposition::Indeterminate,
            COMMITTED => GraphicsCallbackDisposition::Published,
            _ => unreachable!("closed callback request state"),
        }
    }

    pub(super) fn complete(
        &self,
        result: Result<GraphicsCallbackResult, GraphicsCallbackHostFailure>,
    ) {
        let previous =
            self.state
                .compare_exchange(DISPATCHED, COMMITTED, Ordering::AcqRel, Ordering::Acquire);
        debug_assert_eq!(previous, Ok(DISPATCHED));
        if let Some(sender) = self.sender.lock().expect("callback reply lock").take() {
            let _ = sender.send(result);
        }
    }

    /// A registration retired before callback start has no published effect.
    /// A host-ticket Drop racing this rejection wins the same queued CAS; in
    /// either order no callback can later begin for this envelope.
    pub(super) fn reject_queued(&self, failure: GraphicsCallbackHostFailure) {
        if self
            .state
            .compare_exchange(QUEUED, RETIRED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            if let Some(sender) = self.sender.lock().expect("callback reply lock").take() {
                let _ = sender.send(Err(failure));
            }
        }
    }
}

/// A bounded host-owned request acknowledgement. Its queued retirement CAS
/// excludes callback invocation; after dispatch, abandonment is conservatively
/// indeterminate and cannot replay the gesture against another registration.
pub struct GraphicsCallbackHostTicket {
    reply: Arc<HostReply>,
    receiver: oneshot::Receiver<Result<GraphicsCallbackResult, GraphicsCallbackHostFailure>>,
}

impl GraphicsCallbackHostTicket {
    pub fn disposition(&self) -> GraphicsCallbackDisposition {
        self.reply.disposition()
    }

    pub fn abandon(&mut self) -> GraphicsCallbackDisposition {
        let _ =
            self.reply
                .state
                .compare_exchange(QUEUED, RETIRED, Ordering::AcqRel, Ordering::Acquire);
        self.disposition()
    }
}

impl Future for GraphicsCallbackHostTicket {
    type Output = Result<GraphicsCallbackResult, GraphicsCallbackHostFailure>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.receiver).poll(context) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(GraphicsCallbackHostFailure::Retired)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for GraphicsCallbackHostTicket {
    fn drop(&mut self) {
        self.abandon();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_rejection_and_ticket_drop_never_publish() {
        let (reply, ticket) = HostReply::new();
        reply.reject_queued(GraphicsCallbackHostFailure::Retired);
        assert_eq!(
            ticket.disposition(),
            GraphicsCallbackDisposition::NotPublished
        );
        assert_eq!(
            futures::executor::block_on(ticket),
            Err(GraphicsCallbackHostFailure::Retired)
        );

        let (reply, mut ticket) = HostReply::new();
        assert_eq!(ticket.abandon(), GraphicsCallbackDisposition::NotPublished);
        reply.reject_queued(GraphicsCallbackHostFailure::Retired);
        assert_eq!(
            ticket.disposition(),
            GraphicsCallbackDisposition::NotPublished
        );
        assert!(!reply.begin());
    }
}
