use std::sync::{Arc, Mutex};

use futures::future::{select, Either};
use runmat_execution::{cancellation::CancellationSource, CancellationReason};

use crate::host::{CancellationPort, PortFuture};

#[derive(Clone, Debug, Default)]
pub(super) struct InternalCancellation {
    state: Arc<State>,
}

#[derive(Debug, Default)]
struct State {
    source: CancellationSource,
    reason: Mutex<Option<String>>,
}

impl InternalCancellation {
    pub fn cancel(&self, reason: impl Into<String>) {
        let reason = reason.into();
        {
            let mut stored = self
                .state
                .reason
                .lock()
                .expect("cancellation lock poisoned");
            if stored.is_none() {
                *stored = Some(reason);
            }
        }
        // Every caller publishes after storing the first diagnostic, so a
        // concurrent second caller cannot return before notification merely
        // because the first caller paused between these two operations.
        // The shared authority wakes all group listeners outside both locks.
        self.state.source.cancel(CancellationReason::ParentScope);
    }

    fn is_cancelled(&self) -> bool {
        self.state.source.token().is_cancelled()
    }

    fn reason(&self) -> Option<String> {
        if !self.is_cancelled() {
            return None;
        }
        self.state
            .reason
            .lock()
            .expect("cancellation lock poisoned")
            .clone()
    }

    async fn cancelled(&self) -> String {
        self.state.source.token().cancelled().await;
        self.reason()
            .expect("published cancellation retains its diagnostic")
    }
}

#[cfg(test)]
mod tests;

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
