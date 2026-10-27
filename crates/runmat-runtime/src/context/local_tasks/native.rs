use super::driver::{active_driver, ActiveDriverPoll};
use super::*;
mod driver_state;
use driver_state::DriverState;

struct SupervisorInner {
    driver: Rc<DriverState>,
    token: LocalTaskOwnerToken,
    closed: Cell<bool>,
    tasks: RefCell<HashMap<u64, Rc<TaskSlot>>>,
    next_task: Cell<u64>,
    owns_driver: bool,
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
pub(super) struct SubmissionPort {
    supervisor: Weak<SupervisorInner>,
    owner: RuntimeContextLocalId,
    token: LocalTaskOwnerToken,
}

impl RuntimeLocalTaskService for SubmissionPort {
    fn owner_token(&self) -> Option<LocalTaskOwnerToken> {
        Some(self.token.clone())
    }

    fn canonical_admission(&self) -> Option<CanonicalLocalTaskAdmission> {
        Some(CanonicalLocalTaskAdmission { port: self.clone() })
    }

    fn spawn_local(
        &self,
        owner_id: RuntimeContextLocalId,
        future: RuntimeLocalTaskFuture,
    ) -> Result<RuntimeLocalTaskLease, RuntimeError> {
        self.submit(owner_id, future, None)
    }
}

impl SubmissionPort {
    pub(super) fn is_active_driver(&self) -> bool {
        self.supervisor.upgrade().is_some_and(|owner| {
            active_driver()
                .as_ref()
                .is_some_and(|active| active.same(&owner.driver.id))
        })
    }

    pub(super) fn submit(
        &self,
        owner_id: RuntimeContextLocalId,
        future: RuntimeLocalTaskFuture,
        deferred: Option<CommandDeferredWork>,
    ) -> Result<RuntimeLocalTaskLease, RuntimeError> {
        if owner_id != self.owner {
            return reject_admitted(
                LocalTaskFailure::ForeignSession.runtime_error(),
                future,
                deferred,
            );
        }
        let Some(owner) = self.supervisor.upgrade() else {
            return reject_admitted(LocalTaskFailure::Closed.runtime_error(), future, deferred);
        };
        if owner.closed.get() || owner.driver.closed.get() {
            return reject_admitted(LocalTaskFailure::Closed.runtime_error(), future, deferred);
        }
        let local = { owner.driver.local.borrow().as_ref().cloned() };
        let Some(local) = local else {
            return reject_admitted(LocalTaskFailure::Closed.runtime_error(), future, deferred);
        };
        let id = owner.next_task.get();
        let Some(next_id) = id.checked_add(1) else {
            return reject_admitted(
                LocalTaskFailure::IdentityExhausted.runtime_error(),
                future,
                deferred,
            );
        };
        let state = Rc::new(TaskState {
            outcome: Cell::new(None),
            retired: Cell::new(false),
            waiters: RefCell::new(HashMap::new()),
            next_waiter: Cell::new(0),
        });
        owner.next_task.set(next_id);
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
        let slot = Rc::new(TaskSlot::new(child, state.clone(), deferred));
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

/// A weak admission authority for a distinct session on the parent's active
/// local driver. Runtime test-service closures can retain this without keeping
/// either the parent session or its callback supervisor alive.
#[derive(Clone)]
pub struct RuntimeLocalTaskChildFactory(Weak<SupervisorInner>);

impl RuntimeLocalTaskChildFactory {
    pub fn create(&self) -> Result<RuntimeLocalTaskSupervisor, RuntimeError> {
        let parent = self
            .0
            .upgrade()
            .ok_or_else(|| LocalTaskFailure::Closed.runtime_error())?;
        if parent.closed.get() || parent.driver.closed.get() {
            return Err(LocalTaskFailure::Closed.runtime_error());
        }
        let child = Rc::new(SupervisorInner {
            driver: parent.driver.clone(),
            token: LocalTaskOwnerToken::new(),
            closed: Cell::new(false),
            tasks: RefCell::new(HashMap::new()),
            next_task: Cell::new(0),
            owns_driver: false,
        });
        parent
            .driver
            .children
            .borrow_mut()
            .push(Rc::downgrade(&child));
        Ok(RuntimeLocalTaskSupervisor(child))
    }
}

impl RuntimeLocalTaskSupervisor {
    pub fn new() -> Self {
        Self(Rc::new(SupervisorInner {
            driver: Rc::new(DriverState::new()),
            token: LocalTaskOwnerToken::new(),
            closed: Cell::new(false),
            tasks: RefCell::new(HashMap::new()),
            next_task: Cell::new(0),
            owns_driver: true,
        }))
    }

    pub fn child_scope_factory(&self) -> RuntimeLocalTaskChildFactory {
        RuntimeLocalTaskChildFactory(Rc::downgrade(&self.0))
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

    #[cfg(test)]
    pub(super) fn registered_child_count(&self) -> usize {
        self.0.driver.children.borrow().len()
    }

    /// The caller must keep this future polled while awaiting input or another
    /// host event; returning after one foreground call does not drive timers.
    pub async fn drive_until<F: Future>(&self, foreground: F) -> Result<F::Output, RuntimeError> {
        if self.0.closed.get() || self.0.driver.closed.get() {
            return Err(LocalTaskFailure::Closed.runtime_error());
        }
        if active_driver().is_some() {
            return Err(LocalTaskFailure::ReentrantDrive.runtime_error());
        }
        if self.0.driver.driving.replace(true) {
            return Err(LocalTaskFailure::ReentrantDrive.runtime_error());
        }
        struct DriveGuard(Rc<DriverState>);
        impl Drop for DriveGuard {
            fn drop(&mut self) {
                self.0.driving.set(false);
            }
        }
        let _guard = DriveGuard(self.0.driver.clone());
        let local = self
            .0
            .driver
            .local
            .borrow()
            .as_ref()
            .cloned()
            .ok_or_else(|| LocalTaskFailure::Closed.runtime_error())?;
        let run = local.run_until(foreground);
        let mut run = std::pin::pin!(run);
        Ok(std::future::poll_fn(|context| {
            let _active = ActiveDriverPoll::enter(&self.0.driver.id);
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
        if self.0.closed.get() || self.0.driver.closed.get() {
            return Err(LocalTaskFailure::Closed.runtime_error());
        }
        let active = active_driver();
        if active.as_ref().is_some_and(|id| id.same(&self.0.driver.id)) {
            Ok(foreground.await)
        } else if active.is_some() || self.0.driver.driving.get() {
            Err(LocalTaskFailure::ReentrantDrive.runtime_error())
        } else {
            self.drive_until(foreground).await
        }
    }

    pub fn shutdown(&self) {
        if self.0.closed.replace(true) {
            return;
        }
        if self.0.owns_driver {
            self.0.driver.closed.set(true);
            let children = std::mem::take(&mut *self.0.driver.children.borrow_mut());
            for child in children.into_iter().filter_map(|child| child.upgrade()) {
                child.closed.set(true);
                child.cancel_tasks();
            }
        }
        self.0.cancel_tasks();
        if self.0.owns_driver {
            // An active drive retains its own Rc until its poll returns.
            let local = self.0.driver.local.borrow_mut().take();
            drop(local);
        } else {
            self.0.driver.children.borrow_mut().retain(|child| {
                child
                    .upgrade()
                    .is_some_and(|owner| !Rc::ptr_eq(&owner, &self.0))
            });
        }
    }
}

impl SupervisorInner {
    fn cancel_tasks(&self) {
        let tasks = std::mem::take(&mut *self.tasks.borrow_mut());
        for slot in tasks.into_values() {
            slot.cancel();
        }
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
