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
                slot.polling = false;
                slot.cancelled = true;
                slot.wake = None;
                slot.deferred.take()
            };
            drop(deferred);
        }
    }
}

impl TaskSlot {
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
        let (future, wake, deferred) = {
            let mut slot = self.inner.borrow_mut();
            slot.cancelled = true;
            let (future, deferred) = if slot.polling {
                (None, None)
            } else {
                (slot.future.take(), slot.deferred.take())
            };
            (future, slot.wake.take(), deferred)
        };
        self.outcome.finish(RuntimeLocalTaskOutcome::Cancelled);
        drop(future);
        drop(deferred);
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
            slot.polling = false;
            slot.wake = None;
            (slot.cancelled, slot.deferred.take())
        };
        reset.armed = false;
        if let Some(deferred) = deferred {
            deferred.complete(if cancelled {
                DeferredWorkTerminal::Missing
            } else {
                DeferredWorkTerminal::Completed
            });
        }
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
    }
}
