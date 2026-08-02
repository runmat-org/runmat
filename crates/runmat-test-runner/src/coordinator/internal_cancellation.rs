use std::future::poll_fn;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Poll;

use futures::future::{select, Either};
use futures::task::AtomicWaker;

use crate::host::{CancellationPort, PortFuture};

#[derive(Clone, Debug, Default)]
pub(super) struct InternalCancellation {
    state: Arc<State>,
}

#[derive(Debug, Default)]
struct State {
    cancelled: AtomicBool,
    reason: Mutex<Option<String>>,
    waker: AtomicWaker,
}

impl InternalCancellation {
    pub fn cancel(&self, reason: impl Into<String>) {
        let mut stored = self
            .state
            .reason
            .lock()
            .expect("cancellation lock poisoned");
        if stored.is_none() {
            *stored = Some(reason.into());
            self.state.cancelled.store(true, Ordering::Release);
            self.state.waker.wake();
        }
    }

    fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    fn reason(&self) -> Option<String> {
        self.state
            .reason
            .lock()
            .expect("cancellation lock poisoned")
            .clone()
    }

    async fn cancelled(&self) -> String {
        poll_fn(|context| {
            if let Some(reason) = self.reason() {
                return Poll::Ready(reason);
            }
            self.state.waker.register(context.waker());
            if let Some(reason) = self.reason() {
                Poll::Ready(reason)
            } else {
                Poll::Pending
            }
        })
        .await
    }
}

pub(super) struct CombinedCancellation<'a, X> {
    external: &'a X,
    internal: InternalCancellation,
}

impl<'a, X> CombinedCancellation<'a, X> {
    pub fn new(external: &'a X, internal: InternalCancellation) -> Self {
        Self { external, internal }
    }
}

impl<X: CancellationPort> CancellationPort for CombinedCancellation<'_, X> {
    fn is_cancelled(&self) -> bool {
        self.internal.is_cancelled() || self.external.is_cancelled()
    }

    fn reason(&self) -> Option<String> {
        self.internal.reason().or_else(|| self.external.reason())
    }

    fn cancelled<'a>(&'a self) -> PortFuture<'a, String> {
        Box::pin(async move {
            match select(
                Box::pin(self.internal.cancelled()),
                self.external.cancelled(),
            )
            .await
            {
                Either::Left((reason, _)) | Either::Right((reason, _)) => reason,
            }
        })
    }
}
