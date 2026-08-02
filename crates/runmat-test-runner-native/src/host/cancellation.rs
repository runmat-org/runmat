use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use runmat_test_runner::host::{CancellationPort, PortFuture};
use tokio::sync::Notify;

#[derive(Clone, Debug, Default)]
pub struct NativeCancellation {
    state: Arc<CancellationState>,
}

#[derive(Debug, Default)]
struct CancellationState {
    cancelled: AtomicBool,
    reason: Mutex<Option<String>>,
    notify: Notify,
}

impl NativeCancellation {
    pub fn cancel(&self, reason: impl Into<String>) {
        let mut stored = self
            .state
            .reason
            .lock()
            .expect("cancellation lock poisoned");
        if stored.is_none() {
            *stored = Some(reason.into());
            self.state.cancelled.store(true, Ordering::Release);
            self.state.notify.notify_waiters();
        }
    }
}

impl CancellationPort for NativeCancellation {
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

    fn cancelled<'a>(&'a self) -> PortFuture<'a, String> {
        Box::pin(async move {
            loop {
                let notified = self.state.notify.notified();
                if let Some(reason) = self.reason() {
                    return reason;
                }
                notified.await;
            }
        })
    }
}
