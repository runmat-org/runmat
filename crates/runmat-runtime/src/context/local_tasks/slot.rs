//! A cancellable child future owned by the host, independent of executor polling.

use super::{RuntimeLocalTaskFuture, RuntimeLocalTaskOutcome, TaskState};
use std::cell::RefCell;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

struct TaskSlotState {
    future: Option<RuntimeLocalTaskFuture>,
    polling: bool,
    cancelled: bool,
    wake: Option<Waker>,
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
            let mut slot = self.slot.inner.borrow_mut();
            slot.polling = false;
            slot.cancelled = true;
            slot.wake = None;
        }
    }
}

impl TaskSlot {
    pub(super) fn new(future: RuntimeLocalTaskFuture, outcome: Rc<TaskState>) -> Self {
        Self {
            inner: RefCell::new(TaskSlotState {
                future: Some(future),
                polling: false,
                cancelled: false,
                wake: None,
            }),
            outcome,
        }
    }

    pub(super) fn cancel(&self) {
        let (future, wake) = {
            let mut slot = self.inner.borrow_mut();
            slot.cancelled = true;
            let future = if slot.polling {
                None
            } else {
                slot.future.take()
            };
            (future, slot.wake.take())
        };
        self.outcome.finish(RuntimeLocalTaskOutcome::Cancelled);
        drop(future);
        if let Some(wake) = wake {
            wake.wake();
        }
    }

    pub(super) fn poll(&self, context: &mut Context<'_>) -> Poll<()> {
        let mut future = {
            let mut slot = self.inner.borrow_mut();
            if slot.cancelled {
                return Poll::Ready(());
            }
            slot.polling = true;
            slot.wake = Some(context.waker().clone());
            slot.future
                .take()
                .expect("local task polled after completion")
        };
        let mut reset = TaskPollReset {
            slot: self,
            armed: true,
        };
        let result = future.as_mut().poll(context);
        {
            let mut slot = self.inner.borrow_mut();
            slot.polling = false;
            slot.wake = None;
            if result.is_pending() && !slot.cancelled {
                slot.future = Some(future);
                reset.armed = false;
                return Poll::Pending;
            }
        }
        reset.armed = false;
        drop(future);
        Poll::Ready(())
    }
}
