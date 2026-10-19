//! Bounded cross-lane wake for one session-owned callback drain.
//!
//! The GUI owns only `GraphicsCallbackIngress` (Send metadata); this wake
//! stores at most one permit and never captures a RuntimeContext or callable.

use super::*;

impl RuntimeGraphicsCallbackQueue {
    /// Wait for a host admission or owner close. A notification can be
    /// coalesced or spuriously observed after a prior drain, so the caller
    /// must invoke the sole ordered `process()` owner and then wait again.
    /// `false` means this session queue has retired.
    pub(super) async fn wait_for_host_admission(&self) -> bool {
        if self.is_closed() {
            return false;
        }
        self.0.admission_wake.notified().await;
        !self.is_closed()
    }

    /// The session lane must not start an idle drain while a drawnow flush
    /// owns the same ordered queue. A later host wake remains coalesced until
    /// that drain returns, at which point it can be observed again.
    pub(super) fn drain_in_progress(&self) -> bool {
        self.0.draining.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use futures::FutureExt;
    use std::num::NonZeroUsize;

    #[test]
    fn host_admission_wakes_idle_owner_and_close_retires_waiter() {
        futures::executor::block_on(async {
            let queue = RuntimeGraphicsCallbackQueue::new(
                NonZeroUsize::new(2).expect("nonzero pending"),
                NonZeroUsize::new(1).expect("nonzero registrations"),
            );
            let selected = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
            let callback: Rc<GraphicsCallbackResponder> =
                Rc::new(|_| Box::pin(async { Ok(GraphicsCallbackResult::Completed) }));
            let (_owner, ingress) = queue
                .register_response(selected, 9, GraphicsCallbackKind::PanActionPre, callback)
                .expect("register callback");
            let mut waiting = Box::pin(queue.wait_for_host_admission());
            assert!(waiting.as_mut().now_or_never().is_none());
            ingress
                .send(GraphicsCallbackHostEvent {
                    figure: 9,
                    kind: GraphicsCallbackKind::PanActionPre,
                    position: None,
                })
                .expect("host admission");
            assert!(waiting.await);
            queue.process().await.expect("drain admitted callback");

            let mut waiting = Box::pin(queue.wait_for_host_admission());
            assert!(waiting.as_mut().now_or_never().is_none());
            queue.close();
            assert!(!waiting.await);
        });
    }
}
