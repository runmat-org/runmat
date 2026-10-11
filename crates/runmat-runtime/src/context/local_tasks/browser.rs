//! Browser adapter: the JS event loop polls jobs, but this owner can drop an
//! abandoned child immediately without waiting for a later browser callback.

use super::*;
struct BrowserInner {
    token: LocalTaskOwnerToken,
    closed: Cell<bool>,
    tasks: RefCell<HashMap<u64, Rc<TaskSlot>>>,
    next_task: Cell<u64>,
}

struct BrowserRecordOnDrop {
    supervisor: Weak<BrowserInner>,
    id: u64,
}

impl Drop for BrowserRecordOnDrop {
    fn drop(&mut self) {
        if let Some(supervisor) = self.supervisor.upgrade() {
            supervisor.tasks.borrow_mut().remove(&self.id);
        }
    }
}

#[derive(Clone)]
struct BrowserPort {
    supervisor: Weak<BrowserInner>,
    owner: RuntimeContextLocalId,
    token: LocalTaskOwnerToken,
}

impl RuntimeLocalTaskService for BrowserPort {
    fn owner_token(&self) -> Option<LocalTaskOwnerToken> {
        Some(self.token.clone())
    }

    fn spawn_local(
        &self,
        owner_id: RuntimeContextLocalId,
        future: RuntimeLocalTaskFuture,
    ) -> Result<RuntimeLocalTaskLease, RuntimeError> {
        if owner_id != self.owner {
            return Err(LocalTaskFailure::ForeignSession.runtime_error());
        }
        let owner = self
            .supervisor
            .upgrade()
            .ok_or_else(|| LocalTaskFailure::Closed.runtime_error())?;
        if owner.closed.get() {
            return Err(LocalTaskFailure::Closed.runtime_error());
        }
        let state = Rc::new(TaskState {
            outcome: Cell::new(None),
            waiters: RefCell::new(HashMap::new()),
            next_waiter: Cell::new(0),
        });
        let id = owner.next_task.get();
        owner.next_task.set(
            id.checked_add(1)
                .ok_or_else(|| LocalTaskFailure::IdentityExhausted.runtime_error())?,
        );
        let completion = CompletionOnDrop(state.clone());
        let child: RuntimeLocalTaskFuture = Box::pin(async move {
            let completion = completion;
            future.await;
            completion.0.finish(RuntimeLocalTaskOutcome::Completed);
        });
        let slot = Rc::new(TaskSlot::new(child, state.clone()));
        owner.tasks.borrow_mut().insert(id, slot.clone());
        let weak = Rc::downgrade(&owner);
        let abort: Rc<dyn Fn()> = Rc::new(move || {
            let slot = weak
                .upgrade()
                .and_then(|owner| owner.tasks.borrow().get(&id).cloned());
            if let Some(slot) = slot {
                slot.cancel();
            }
        });
        let weak = Rc::downgrade(&owner);
        wasm_bindgen_futures::spawn_local(async move {
            let _record = BrowserRecordOnDrop {
                supervisor: weak,
                id,
            };
            std::future::poll_fn(|context| slot.poll(context)).await;
        });
        Ok(RuntimeLocalTaskLease { state, abort })
    }
}

/// Host-held browser task owner. JavaScript supplies the idle event loop.
pub struct BrowserLocalTaskSupervisor(Rc<BrowserInner>);

impl BrowserLocalTaskSupervisor {
    pub fn new() -> Self {
        Self(Rc::new(BrowserInner {
            token: LocalTaskOwnerToken::new(),
            closed: Cell::new(false),
            tasks: RefCell::new(HashMap::new()),
            next_task: Cell::new(0),
        }))
    }

    pub fn submission_port(&self, owner: RuntimeContextLocalId) -> Rc<dyn RuntimeLocalTaskService> {
        Rc::new(BrowserPort {
            supervisor: Rc::downgrade(&self.0),
            owner,
            token: self.0.token.clone(),
        })
    }

    pub fn owner_token(&self) -> LocalTaskOwnerToken {
        self.0.token.clone()
    }

    pub fn shutdown(&self) {
        if self.0.closed.replace(true) {
            return;
        }
        let tasks = std::mem::take(&mut *self.0.tasks.borrow_mut());
        for slot in tasks.into_values() {
            slot.cancel();
        }
    }
}

impl Default for BrowserLocalTaskSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for BrowserLocalTaskSupervisor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RuntimeContext;
    use crate::execution::RuntimeExecutionService;
    use std::marker::PhantomPinned;
    use std::task::{Context, Poll};
    use wasm_bindgen_test::wasm_bindgen_test;

    fn owner() -> RuntimeContextLocalId {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).local_identity()
    }

    #[wasm_bindgen_test(async)]
    async fn browser_event_loop_completes_local_non_send_task() {
        let supervisor = BrowserLocalTaskSupervisor::new();
        let owner = owner();
        let count = Rc::new(Cell::new(0));
        let child_count = count.clone();
        let lease = supervisor
            .submission_port(owner)
            .spawn_local(
                owner,
                Box::pin(async move {
                    child_count.set(1);
                }),
            )
            .expect("spawn");
        assert_eq!(lease.join().await, RuntimeLocalTaskOutcome::Completed);
        assert_eq!(count.get(), 1);
    }

    #[wasm_bindgen_test(async)]
    async fn browser_shutdown_from_current_callback_reports_cancelled() {
        let supervisor = Rc::new(BrowserLocalTaskSupervisor::new());
        let weak = Rc::downgrade(&supervisor);
        let owner = owner();
        let lease = supervisor
            .submission_port(owner)
            .spawn_local(
                owner,
                Box::pin(async move {
                    weak.upgrade().expect("host owns supervisor").shutdown();
                    // The child reaches Ready in the same poll that requested shutdown.
                }),
            )
            .expect("spawn");
        assert_eq!(lease.join().await, RuntimeLocalTaskOutcome::Cancelled);
        assert_eq!(lease.outcome(), Some(RuntimeLocalTaskOutcome::Cancelled));
    }

    struct PendingDrop(Rc<Cell<usize>>, PhantomPinned);

    impl Future for PendingDrop {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    impl Drop for PendingDrop {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[wasm_bindgen_test]
    fn browser_shutdown_drops_unpolled_pinned_child_synchronously() {
        let supervisor = BrowserLocalTaskSupervisor::new();
        let owner = owner();
        let drops = Rc::new(Cell::new(0));
        let lease = supervisor
            .submission_port(owner)
            .spawn_local(owner, Box::pin(PendingDrop(drops.clone(), PhantomPinned)))
            .expect("spawn");
        supervisor.shutdown();
        assert_eq!(drops.get(), 1);
        assert_eq!(lease.outcome(), Some(RuntimeLocalTaskOutcome::Cancelled));
    }
}
