//! A cancellable child future owned by the host, independent of executor polling.

use super::{RuntimeLocalTaskFuture, RuntimeLocalTaskOutcome, TaskState};
use runmat_execution::profiling::{CommandDeferredWork, DeferredWorkTerminal};
use std::cell::RefCell;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

struct TaskSlotState {
    future: Option<RuntimeLocalTaskFuture>,
    polling: bool,
    cancelled: bool,
    wake: Option<Waker>,
    deferred: Option<CommandDeferredWork>,
}

pub(super) struct TaskSlot {
    inner: RefCell<TaskSlotState>,
    outcome: Rc<TaskState>,
}

struct TaskPollReset<'a> {
    slot: &'a TaskSlot,
    armed: bool,
}

impl Drop for TaskPollReset<'_> {
    fn drop(&mut self) {
        if self.armed {
            let deferred = {
                let mut slot = self.slot.inner.borrow_mut();
                slot.cancelled = true;
                slot.wake = None;
                slot.deferred.take()
            };
            drop(deferred);
            self.slot.inner.borrow_mut().polling = false;
            self.slot.outcome.finish(RuntimeLocalTaskOutcome::Cancelled);
            self.slot.outcome.retire();
        }
    }
}

impl TaskSlot {
    #[cfg(target_arch = "wasm32")]
    pub(super) fn is_retired(&self) -> bool {
        self.outcome.retired.get()
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) async fn wait_for_retirement(&self) {
        super::lease::wait_for_retirement(self.outcome.clone()).await;
    }

    pub(super) fn new(
        future: RuntimeLocalTaskFuture,
        outcome: Rc<TaskState>,
        deferred: Option<CommandDeferredWork>,
    ) -> Self {
        Self {
            inner: RefCell::new(TaskSlotState {
                future: Some(future),
                polling: false,
                cancelled: false,
                wake: None,
                deferred,
            }),
            outcome,
        }
    }

    pub(super) fn cancel(&self) {
        let (future, wake, deferred, polling) = {
            let mut slot = self.inner.borrow_mut();
            slot.cancelled = true;
            let polling = slot.polling;
            let (future, deferred) = if polling {
                (None, None)
            } else {
                // Destructors can reenter cancellation. Keep the slot active
                // until both owned payloads have actually finished dropping.
                slot.polling = true;
                (slot.future.take(), slot.deferred.take())
            };
            (future, slot.wake.take(), deferred, polling)
        };
        self.outcome.finish(RuntimeLocalTaskOutcome::Cancelled);
        drop(future);
        drop(deferred);
        if !polling {
            self.inner.borrow_mut().polling = false;
            self.outcome.retire();
        }
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    pub(super) fn poll(&self, context: &mut Context<'_>) -> Poll<()> {
        {
            let mut slot = self.inner.borrow_mut();
            if slot.cancelled {
                return Poll::Ready(());
            }
            slot.polling = true;
            slot.wake = Some(context.waker().clone());
        }
        // Declared before the future: unwinding destroys the future first,
        // then this guard settles the deferred owner as Missing.
        let mut reset = TaskPollReset {
            slot: self,
            armed: true,
        };
        let mut future = self
            .inner
            .borrow_mut()
            .future
            .take()
            .expect("local task polled after completion");
        let result = future.as_mut().poll(context);
        {
            let mut slot = self.inner.borrow_mut();
            if result.is_pending() && !slot.cancelled {
                slot.future = Some(future);
                slot.polling = false;
                slot.wake = None;
                reset.armed = false;
                return Poll::Pending;
            }
        }
        // Keep polling=true through Drop: a reentrant cancel from a captured
        // value's destructor may mark cancellation but cannot settle early.
        drop(future);
        let (cancelled, deferred) = {
            let mut slot = self.inner.borrow_mut();
            slot.wake = None;
            (slot.cancelled, slot.deferred.take())
        };
        if let Some(deferred) = deferred {
            deferred.complete(if cancelled {
                DeferredWorkTerminal::Missing
            } else {
                DeferredWorkTerminal::Completed
            });
        }
        self.inner.borrow_mut().polling = false;
        reset.armed = false;
        self.outcome.retire();
        Poll::Ready(())
    }
}

impl Drop for TaskSlot {
    fn drop(&mut self) {
        let (future, deferred) = {
            let state = self.inner.get_mut();
            (state.future.take(), state.deferred.take())
        };
        drop(future);
        drop(deferred);
        self.outcome.finish(RuntimeLocalTaskOutcome::Cancelled);
        self.outcome.retire();
    }
}
