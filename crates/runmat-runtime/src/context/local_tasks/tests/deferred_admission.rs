use super::*;
use crate::context::LocalTaskCommandOrigin;
use runmat_execution::profiling::{CommandProfilingRoot, DeferredWorkCounts};
use std::cell::RefCell;
use std::collections::HashMap;

fn profiled_runtime(
    supervisor: &RuntimeLocalTaskSupervisor,
    command: &CommandProfilingRoot,
) -> RuntimeContext {
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let ports = RuntimeServicePorts::default()
        .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
    runtime
        .with_service_ports(ports)
        .with_command_profiling(command.receipt())
}

fn deferred_counts(command: CommandProfilingRoot) -> DeferredWorkCounts {
    command
        .seal()
        .deferred_work
        .expect("valid deferred receipt")
}

struct DelegatingPort {
    token: LocalTaskOwnerToken,
    canonical: Option<CanonicalLocalTaskAdmission>,
    raw: Rc<dyn RuntimeLocalTaskService>,
    raw_calls: Rc<Cell<usize>>,
}

impl RuntimeLocalTaskService for DelegatingPort {
    fn owner_token(&self) -> Option<LocalTaskOwnerToken> {
        Some(self.token.clone())
    }

    fn canonical_admission(&self) -> Option<CanonicalLocalTaskAdmission> {
        self.canonical.clone()
    }

    fn spawn_local(
        &self,
        owner: RuntimeContextLocalId,
        future: RuntimeLocalTaskFuture,
    ) -> Result<RuntimeLocalTaskLease, RuntimeError> {
        self.raw_calls.set(self.raw_calls.get() + 1);
        self.raw.spawn_local(owner, future)
    }
}

#[test]
fn cloned_owner_token_cannot_certify_custom_tasking_but_real_capability_can() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let base = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let actual = supervisor.submission_port(base.local_identity());
    let token = actual.owner_token().expect("canonical owner token");
    let real_capability = actual.canonical_admission().expect("canonical capability");
    let raw_calls = Rc::new(Cell::new(0));

    let forged_command = CommandProfilingRoot::new();
    let forged_port = Rc::new(DelegatingPort {
        token: token.clone(),
        canonical: None,
        raw: actual.clone(),
        raw_calls: raw_calls.clone(),
    });
    let forged_runtime = base
        .clone()
        .with_service_ports(RuntimeServicePorts::default().with_local_tasks(forged_port))
        .with_command_profiling(forged_command.receipt());
    let forged = forged_runtime
        .spawn_local_with_command_origin(
            "custom task",
            LocalTaskCommandOrigin::CurrentSelected,
            |selected| {
                assert!(selected.command_profiling().is_none());
                Box::pin(async {})
            },
        )
        .expect("custom semantic task still runs");
    host()
        .block_on(supervisor.drive_until(forged.join()))
        .expect("host drive");
    assert_eq!(raw_calls.get(), 1);
    assert_eq!(deferred_counts(forged_command).admitted, 0);

    let delegated_command = CommandProfilingRoot::new();
    let delegated_port = Rc::new(DelegatingPort {
        token,
        canonical: Some(real_capability),
        raw: actual,
        raw_calls: raw_calls.clone(),
    });
    let delegated_runtime = base
        .with_service_ports(RuntimeServicePorts::default().with_local_tasks(delegated_port))
        .with_command_profiling(delegated_command.receipt());
    let delegated = delegated_runtime
        .spawn_local_with_command_origin(
            "delegated canonical task",
            LocalTaskCommandOrigin::CurrentSelected,
            |selected| {
                assert!(selected.command_profiling().is_some());
                Box::pin(async {})
            },
        )
        .expect("delegated canonical admission");
    host()
        .block_on(supervisor.drive_until(delegated.join()))
        .expect("host drive");
    assert_eq!(
        raw_calls.get(),
        1,
        "delegation must bypass custom raw scheduling"
    );
    assert_eq!(deferred_counts(delegated_command).completed, 1);
}

#[test]
fn unknown_origin_masks_scoped_and_captured_task_context() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let command = CommandProfilingRoot::new();
    let runtime = profiled_runtime(&supervisor, &command);
    let lease = runtime
        .spawn_local_with_command_origin(
            "permanent idle service",
            LocalTaskCommandOrigin::Unknown,
            |selected| {
                assert!(selected.command_profiling().is_none());
                let captured = selected.clone();
                Box::pin(selected.scope(async move {
                    assert!(captured.command_profiling().is_none());
                    assert!(crate::context::scope::active_runtime_context()
                        .and_then(|active| active.command_profiling().cloned())
                        .is_none());
                }))
            },
        )
        .expect("unknown task admission");
    host()
        .block_on(supervisor.drive_until(lease.join()))
        .expect("host drive");
    assert_eq!(deferred_counts(command).admitted, 0);
}

#[test]
fn canonical_admission_retires_future_before_completed_receipt() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let command = CommandProfilingRoot::new();
    let runtime = profiled_runtime(&supervisor, &command);
    let drops = Rc::new(Cell::new(0));
    let task_drops = drops.clone();
    let lease = runtime
        .spawn_local_with_command_origin(
            "profiled task",
            LocalTaskCommandOrigin::CurrentSelected,
            move |selected| {
                assert!(selected.command_profiling().is_some());
                Box::pin(async move {
                    let _captured = PendingDrop {
                        drops: task_drops,
                        _pinned: PhantomPinned,
                    };
                })
            },
        )
        .expect("canonical admission");
    let outcome = host()
        .block_on(supervisor.drive_until(lease.join()))
        .expect("host drive");
    assert_eq!(outcome, RuntimeLocalTaskOutcome::Completed);
    assert_eq!(drops.get(), 1);
    assert_eq!(
        deferred_counts(command),
        DeferredWorkCounts {
            admitted: 1,
            pending: 0,
            completed: 1,
            failed: 0,
            missing: 0,
        }
    );
}

#[test]
fn unpolled_cancel_retires_future_before_missing_receipt() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let command = CommandProfilingRoot::new();
    let runtime = profiled_runtime(&supervisor, &command);
    let drops = Rc::new(Cell::new(0));
    let task_drops = drops.clone();
    let lease = runtime
        .spawn_local_with_command_origin(
            "profiled task",
            LocalTaskCommandOrigin::CurrentSelected,
            move |_| {
                Box::pin(PendingDrop {
                    drops: task_drops,
                    _pinned: PhantomPinned,
                })
            },
        )
        .expect("canonical admission");
    lease.cancel();
    assert_eq!(drops.get(), 1);
    assert_eq!(lease.outcome(), Some(RuntimeLocalTaskOutcome::Cancelled));
    assert_eq!(
        deferred_counts(command),
        DeferredWorkCounts {
            admitted: 1,
            pending: 0,
            completed: 0,
            failed: 0,
            missing: 1,
        }
    );
}

#[test]
fn active_cancel_wins_same_poll_ready_for_deferred_receipt() {
    let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
    let command = CommandProfilingRoot::new();
    let runtime = profiled_runtime(&supervisor, &command);
    let weak = Rc::downgrade(&supervisor);
    let (sender, receiver) = futures::channel::oneshot::channel();
    let lease = runtime
        .spawn_local_with_command_origin(
            "profiled task",
            LocalTaskCommandOrigin::CurrentSelected,
            move |_| {
                Box::pin(async move {
                    weak.upgrade().expect("host alive").shutdown();
                    let _ = sender.send(());
                })
            },
        )
        .expect("canonical admission");
    host()
        .block_on(supervisor.drive_until(async {
            receiver.await.expect("task signalled after shutdown");
        }))
        .expect("same drive exits");
    assert_eq!(lease.outcome(), Some(RuntimeLocalTaskOutcome::Cancelled));
    assert_eq!(deferred_counts(command).missing, 1);
}

#[test]
fn rejected_canonical_admission_drops_future_before_failed_receipt() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let port_owner = owner();
    let foreign = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let command = CommandProfilingRoot::new();
    let runtime = foreign
        .with_service_ports(
            RuntimeServicePorts::default().with_local_tasks(supervisor.submission_port(port_owner)),
        )
        .with_command_profiling(command.receipt());
    let drops = Rc::new(Cell::new(0));
    let task_drops = drops.clone();
    let result = runtime.spawn_local_with_command_origin(
        "foreign canonical task",
        LocalTaskCommandOrigin::CurrentSelected,
        move |_| {
            Box::pin(PendingDrop {
                drops: task_drops,
                _pinned: PhantomPinned,
            })
        },
    );
    assert!(result.is_err());
    assert_eq!(drops.get(), 1);
    assert_eq!(
        deferred_counts(command),
        DeferredWorkCounts {
            admitted: 1,
            pending: 0,
            completed: 0,
            failed: 1,
            missing: 0,
        }
    );
}

#[test]
fn sealed_origin_does_not_transfer_deferred_ownership_to_later_driver() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let first = CommandProfilingRoot::new();
    let runtime = profiled_runtime(&supervisor, &first);
    let (sender, receiver) = futures::channel::oneshot::channel();
    let lease = runtime
        .spawn_local_with_command_origin(
            "first command task",
            LocalTaskCommandOrigin::CurrentSelected,
            move |selected| {
                Box::pin(selected.clone().scope(async move {
                    let _ = sender.send(());
                    std::future::pending::<()>().await;
                }))
            },
        )
        .expect("first admission");
    let snapshot = first.seal();
    assert_eq!(snapshot.deferred_work.expect("valid").admitted, 1);
    assert_eq!(snapshot.deferred_work.expect("valid").pending, 1);
    let second = CommandProfilingRoot::new();
    host()
        .block_on(supervisor.drive_until(async {
            receiver
                .await
                .expect("first task polled under later host drive");
        }))
        .expect("later drive");
    lease.cancel();
    assert_eq!(deferred_counts(second).admitted, 0);
}

#[test]
fn child_supervisor_task_uses_its_selected_command_origin() {
    let parent = RuntimeLocalTaskSupervisor::new();
    let child = parent.child_scope_factory().create().expect("child owner");
    let command = CommandProfilingRoot::new();
    let runtime = profiled_runtime(&child, &command);
    let lease = runtime
        .spawn_local_with_command_origin(
            "child task",
            LocalTaskCommandOrigin::CurrentSelected,
            |selected| {
                assert!(selected.command_profiling().is_some());
                Box::pin(async {})
            },
        )
        .expect("child admission");
    assert_eq!(
        host()
            .block_on(parent.drive_until(lease.join()))
            .expect("parent drives child"),
        RuntimeLocalTaskOutcome::Completed
    );
    assert_eq!(deferred_counts(command).completed, 1);
}

struct PanicOnPoll(Rc<Cell<usize>>);

impl Future for PanicOnPoll {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        panic!("owned future panicked during poll")
    }
}

struct SealOnDrop {
    command: Option<CommandProfilingRoot>,
    observed: Rc<Cell<Option<DeferredWorkCounts>>>,
}

impl Future for SealOnDrop {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(())
    }
}

impl Drop for SealOnDrop {
    fn drop(&mut self) {
        let command = self.command.take().expect("one terminal seal");
        self.observed.set(command.seal().deferred_work);
    }
}

#[test]
fn captured_future_drop_observes_pending_before_task_terminal_settlement() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let command = CommandProfilingRoot::new();
    let runtime = profiled_runtime(&supervisor, &command);
    let observed = Rc::new(Cell::new(None));
    let terminal = observed.clone();
    let lease = runtime
        .spawn_local_with_command_origin(
            "drop-order witness",
            LocalTaskCommandOrigin::CurrentSelected,
            move |_| {
                Box::pin(SealOnDrop {
                    command: Some(command),
                    observed: terminal,
                })
            },
        )
        .expect("canonical admission");
    assert_eq!(
        host()
            .block_on(supervisor.drive_until(lease.join()))
            .expect("host drive"),
        RuntimeLocalTaskOutcome::Completed
    );
    let at_drop = observed.get().expect("future was destroyed");
    assert_eq!(at_drop.admitted, 1);
    assert_eq!(
        at_drop.pending, 1,
        "future Drop must precede deferred settlement"
    );
    assert_eq!(at_drop.completed, 0);
}

impl Drop for PanicOnPoll {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn unwind_drops_owned_future_before_deferred_missing_settlement() {
    let command = CommandProfilingRoot::new();
    let deferred = command
        .receipt()
        .begin_deferred_work()
        .expect("admitted deferred owner");
    let state = Rc::new(TaskState {
        outcome: Cell::new(None),
        retired: Cell::new(false),
        waiters: RefCell::new(HashMap::new()),
        next_waiter: Cell::new(0),
    });
    let drops = Rc::new(Cell::new(0));
    let slot = TaskSlot::new(Box::pin(PanicOnPoll(drops.clone())), state, Some(deferred));
    let wake = futures::task::noop_waker();
    let mut context = Context::from_waker(&wake);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| slot.poll(&mut context)));
    assert!(result.is_err());
    assert_eq!(drops.get(), 1);
    assert_eq!(deferred_counts(command).missing, 1);
}

#[test]
fn cancellation_retirement_fence_follows_reentrant_future_drop() {
    struct ReentrantDrop {
        slot: Rc<RefCell<std::rc::Weak<TaskSlot>>>,
        state: Rc<TaskState>,
        observed: Rc<Cell<bool>>,
    }
    impl Future for ReentrantDrop {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            Poll::Pending
        }
    }
    impl Drop for ReentrantDrop {
        fn drop(&mut self) {
            let slot = self.slot.borrow().upgrade().expect("slot is live");
            slot.cancel();
            assert_eq!(
                self.state.outcome.get(),
                Some(RuntimeLocalTaskOutcome::Cancelled)
            );
            assert!(
                !self.state.retired.get(),
                "reentrant cancellation is not a drop fence"
            );
            self.observed.set(true);
        }
    }
    let state = Rc::new(TaskState {
        outcome: Cell::new(None),
        retired: Cell::new(false),
        waiters: RefCell::new(HashMap::new()),
        next_waiter: Cell::new(0),
    });
    let weak = Rc::new(RefCell::new(std::rc::Weak::new()));
    let observed = Rc::new(Cell::new(false));
    let slot = Rc::new(TaskSlot::new(
        Box::pin(ReentrantDrop {
            slot: weak.clone(),
            state: state.clone(),
            observed: observed.clone(),
        }),
        state.clone(),
        None,
    ));
    *weak.borrow_mut() = Rc::downgrade(&slot);
    slot.cancel();
    assert!(observed.get());
    assert!(state.retired.get());
}
