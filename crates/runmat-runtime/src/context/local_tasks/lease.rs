use super::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::task::{Poll, Waker};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeLocalTaskOutcome {
    Completed,
    Cancelled,
}

pub(super) struct TaskState {
    pub(super) outcome: Cell<Option<RuntimeLocalTaskOutcome>>,
    pub(super) waiters: RefCell<HashMap<u64, Waker>>,
    pub(super) next_waiter: Cell<u64>,
}

impl TaskState {
    pub(super) fn finish(&self, outcome: RuntimeLocalTaskOutcome) {
        if self.outcome.get().is_some() {
            return;
        }
        self.outcome.set(Some(outcome));
        let waiters = std::mem::take(&mut *self.waiters.borrow_mut());
        for waiter in waiters.into_values() {
            waiter.wake();
        }
    }
}

/// Cancelling is idempotent. Dropping a lease does not cancel a running task;
/// its owner must retain the lease until completion or explicit shutdown.
#[derive(Clone)]
pub struct RuntimeLocalTaskLease {
    pub(super) state: Rc<TaskState>,
    pub(super) abort: Rc<dyn Fn()>,
}

impl RuntimeLocalTaskLease {
    pub fn cancel(&self) {
        if self.state.outcome.get().is_none() {
            (self.abort)();
        }
    }

    pub fn outcome(&self) -> Option<RuntimeLocalTaskOutcome> {
        self.state.outcome.get()
    }

    pub fn join(&self) -> impl Future<Output = RuntimeLocalTaskOutcome> {
        JoinWaiter {
            state: self.state.clone(),
            registration: None,
        }
    }
}

struct JoinWaiter {
    state: Rc<TaskState>,
    registration: Option<u64>,
}

impl Future for JoinWaiter {
    type Output = RuntimeLocalTaskOutcome;

    fn poll(mut self: Pin<&mut Self>, context: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        if let Some(outcome) = self.state.outcome.get() {
            return Poll::Ready(outcome);
        }
        let id = match self.registration {
            Some(id) => id,
            None => {
                let start = self.state.next_waiter.get();
                let mut id = start;
                while self.state.waiters.borrow().contains_key(&id) {
                    id = id.wrapping_add(1);
                    if id == start {
                        return Poll::Pending;
                    }
                }
                self.state.next_waiter.set(id.wrapping_add(1));
                self.registration = Some(id);
                id
            }
        };
        self.state
            .waiters
            .borrow_mut()
            .insert(id, context.waker().clone());
        Poll::Pending
    }
}

impl Drop for JoinWaiter {
    fn drop(&mut self) {
        if let Some(id) = self.registration.take() {
            self.state.waiters.borrow_mut().remove(&id);
        }
    }
}

pub(super) struct CompletionOnDrop(pub(super) Rc<TaskState>);

impl Drop for CompletionOnDrop {
    fn drop(&mut self) {
        self.0.finish(RuntimeLocalTaskOutcome::Cancelled);
    }
}
