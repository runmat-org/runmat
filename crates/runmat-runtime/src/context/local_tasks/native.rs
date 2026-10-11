use super::driver::{active_driver, ActiveDriverPoll, SupervisorId};
use super::*;
use tokio::task::LocalSet;

struct SupervisorInner {
    id: SupervisorId,
    token: LocalTaskOwnerToken,
    local: RefCell<Option<Rc<LocalSet>>>,
    closed: Cell<bool>,
    driving: Cell<bool>,
    tasks: RefCell<HashMap<u64, Rc<TaskSlot>>>,
    next_task: Cell<u64>,
}

struct NativeCompletionOnDrop {
    state: Rc<TaskState>,
    supervisor: Weak<SupervisorInner>,
    id: u64,
}

impl Drop for NativeCompletionOnDrop {
    fn drop(&mut self) {
        self.state.finish(RuntimeLocalTaskOutcome::Cancelled);
        if let Some(supervisor) = self.supervisor.upgrade() {
            supervisor.tasks.borrow_mut().remove(&self.id);
        }
    }
}

#[derive(Clone)]
struct SubmissionPort {
    supervisor: Weak<SupervisorInner>,
    owner: RuntimeContextLocalId,
    token: LocalTaskOwnerToken,
}

impl RuntimeLocalTaskService for SubmissionPort {
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
        let local = owner
            .local
            .borrow()
            .as_ref()
            .cloned()
            .ok_or_else(|| LocalTaskFailure::Closed.runtime_error())?;
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
        let completion = NativeCompletionOnDrop {
            state: state.clone(),
            supervisor: Rc::downgrade(&owner),
            id,
        };
        let child: RuntimeLocalTaskFuture = Box::pin(async move {
            let completion = completion;
            future.await;
            completion.state.finish(RuntimeLocalTaskOutcome::Completed);
        });
        let slot = Rc::new(TaskSlot::new(child, state.clone()));
        let task_slot = slot.clone();
        let _task = local.spawn_local(async move {
            std::future::poll_fn(|context| task_slot.poll(context)).await;
        });
        {
            let mut tasks = owner.tasks.borrow_mut();
            tasks.insert(id, slot);
        }
        let weak = Rc::downgrade(&owner);
        let cancel: Rc<dyn Fn()> = Rc::new(move || {
            let slot = weak
                .upgrade()
                .and_then(|owner| owner.tasks.borrow().get(&id).cloned());
            if let Some(slot) = slot {
                slot.cancel();
            }
        });
        Ok(RuntimeLocalTaskLease {
            state,
            abort: cancel,
        })
    }
}

/// The host, not RuntimeContext, retains this owner for the session lifetime.
pub struct RuntimeLocalTaskSupervisor(Rc<SupervisorInner>);

impl RuntimeLocalTaskSupervisor {
    pub fn new() -> Self {
        Self(Rc::new(SupervisorInner {
            id: SupervisorId::new(),
            token: LocalTaskOwnerToken::new(),
            local: RefCell::new(Some(Rc::new(LocalSet::new()))),
            closed: Cell::new(false),
            driving: Cell::new(false),
            tasks: RefCell::new(HashMap::new()),
            next_task: Cell::new(0),
        }))
    }

    pub fn submission_port(&self, owner: RuntimeContextLocalId) -> Rc<dyn RuntimeLocalTaskService> {
        Rc::new(SubmissionPort {
            supervisor: Rc::downgrade(&self.0),
            owner,
            token: self.0.token.clone(),
        })
    }

    pub fn owner_token(&self) -> LocalTaskOwnerToken {
        self.0.token.clone()
    }

    #[cfg(test)]
    pub(super) fn retained_task_count(&self) -> usize {
        self.0.tasks.borrow().len()
    }

    /// The caller must keep this future polled while awaiting input or another
    /// host event; returning after one foreground call does not drive timers.
    pub async fn drive_until<F: Future>(&self, foreground: F) -> Result<F::Output, RuntimeError> {
        if self.0.closed.get() {
            return Err(LocalTaskFailure::Closed.runtime_error());
        }
        if active_driver().is_some() {
            return Err(LocalTaskFailure::ReentrantDrive.runtime_error());
        }
        if self.0.driving.replace(true) {
            return Err(LocalTaskFailure::ReentrantDrive.runtime_error());
        }
        struct DriveGuard(Rc<SupervisorInner>);
        impl Drop for DriveGuard {
            fn drop(&mut self) {
                self.0.driving.set(false);
            }
        }
        let _guard = DriveGuard(self.0.clone());
        let local = self
            .0
            .local
            .borrow()
            .as_ref()
            .cloned()
            .ok_or_else(|| LocalTaskFailure::Closed.runtime_error())?;
        let run = local.run_until(foreground);
        let mut run = std::pin::pin!(run);
        Ok(std::future::poll_fn(|context| {
            let _active = ActiveDriverPoll::enter(&self.0.id);
            run.as_mut().poll(context)
        })
        .await)
    }

    /// Core's foreground entrypoints participate in an existing host idle
    /// drive, or own a drive for a direct library invocation. This avoids
    /// recursively entering LocalSet when a REPL event launches a command.
    pub async fn participate_foreground<F: Future>(
        &self,
        foreground: F,
    ) -> Result<F::Output, RuntimeError> {
        if self.0.closed.get() {
            return Err(LocalTaskFailure::Closed.runtime_error());
        }
        let active = active_driver();
        if active.as_ref().is_some_and(|id| id.same(&self.0.id)) {
            Ok(foreground.await)
        } else if active.is_some() || self.0.driving.get() {
            Err(LocalTaskFailure::ReentrantDrive.runtime_error())
        } else {
            self.drive_until(foreground).await
        }
    }

    pub fn shutdown(&self) {
        if self.0.closed.replace(true) {
            return;
        }
        let tasks = std::mem::take(&mut *self.0.tasks.borrow_mut());
        for slot in tasks.into_values() {
            slot.cancel();
        }
        // No borrow is held while the LocalSet destroys pending callback
        // futures. An in-flight drive retains its own Rc until that poll exits.
        let local = self.0.local.borrow_mut().take();
        drop(local);
    }
}

impl Default for RuntimeLocalTaskSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RuntimeLocalTaskSupervisor {
    fn drop(&mut self) {
        self.shutdown();
    }
}
