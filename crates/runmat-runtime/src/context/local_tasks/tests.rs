use super::*;
use crate::context::{RuntimeContext, RuntimeServicePorts};
use crate::execution::RuntimeExecutionService;
use std::marker::PhantomPinned;
use std::task::{Context, Poll};

struct PendingDrop {
    drops: Rc<Cell<usize>>,
    _pinned: PhantomPinned,
}

impl Future for PendingDrop {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Pending
    }
}

impl Drop for PendingDrop {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

fn host() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test host reactor")
}

fn owner() -> RuntimeContextLocalId {
    RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).local_identity()
}

#[test]
fn local_non_send_task_progresses_while_host_drives_idle_wait() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let owner = owner();
    let port = supervisor.submission_port(owner);
    let counter = Rc::new(Cell::new(0));
    let local = counter.clone();
    let lease = port
        .spawn_local(
            owner,
            Box::pin(async move {
                tokio::task::yield_now().await;
                local.set(local.get() + 1);
            }),
        )
        .expect("spawn");
    assert_eq!(counter.get(), 0);
    let result = host()
        .block_on(supervisor.drive_until(lease.join()))
        .expect("idle driver");
    assert_eq!(result, RuntimeLocalTaskOutcome::Completed);
    assert_eq!(counter.get(), 1);
    supervisor.shutdown();
    assert!(port.spawn_local(owner, Box::pin(async {})).is_err());
}

#[test]
fn foreign_session_cannot_submit_to_a_cloned_port() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let owner_id = owner();
    let other = owner();
    let port = supervisor.submission_port(owner_id);
    assert!(port.spawn_local(other, Box::pin(async {})).is_err());
    assert!(port.spawn_local(owner_id, Box::pin(async {})).is_ok());
    supervisor.shutdown();
}

#[test]
fn multiple_waiters_observe_one_completion_without_consuming_it() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let owner = owner();
    let lease = supervisor
        .submission_port(owner)
        .spawn_local(owner, Box::pin(async { tokio::task::yield_now().await }))
        .expect("spawn");
    let (first, second) = host()
        .block_on(supervisor.drive_until(async { futures::join!(lease.join(), lease.join()) }))
        .expect("drive both waiters");
    assert_eq!(first, RuntimeLocalTaskOutcome::Completed);
    assert_eq!(second, RuntimeLocalTaskOutcome::Completed);
    assert_eq!(lease.outcome(), Some(RuntimeLocalTaskOutcome::Completed));
}

#[test]
fn abandoned_join_removes_its_waker_registration() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let owner = owner();
    let lease = supervisor
        .submission_port(owner)
        .spawn_local(owner, Box::pin(std::future::pending()))
        .expect("spawn");
    let mut join = Box::pin(lease.join());
    let wake = futures::task::noop_waker();
    let mut context = Context::from_waker(&wake);
    assert!(join.as_mut().poll(&mut context).is_pending());
    assert_eq!(lease.state.waiters.borrow().len(), 1);
    drop(join);
    assert!(lease.state.waiters.borrow().is_empty());
    supervisor.shutdown();
}

#[test]
fn nested_foreground_driver_is_rejected_without_entering_a_second_local_set() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let inner = host()
        .block_on(supervisor.drive_until(async { supervisor.drive_until(async { 1usize }).await }))
        .expect("outer drive");
    assert!(inner.is_err());
    assert_eq!(
        host()
            .block_on(supervisor.drive_until(async { 2usize }))
            .unwrap(),
        2
    );
}

#[test]
fn direct_foreground_drive_needs_no_tokio_reactor_for_local_work() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let owner = owner();
    let observed = Rc::new(Cell::new(false));
    let callback_observed = observed.clone();
    let lease = supervisor
        .submission_port(owner)
        .spawn_local(owner, Box::pin(async move { callback_observed.set(true) }))
        .expect("spawn without host reactor");
    let outcome = futures::executor::block_on(supervisor.participate_foreground(lease.join()))
        .expect("local task drive without Tokio reactor");
    assert_eq!(outcome, RuntimeLocalTaskOutcome::Completed);
    assert!(observed.get());
    assert!(host().block_on(supervisor.drive_until(async {})).is_ok());
}

#[test]
fn foreground_participates_without_reentering_host_idle_driver() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let result = host()
        .block_on(
            supervisor
                .drive_until(async { supervisor.participate_foreground(async { 7usize }).await }),
        )
        .expect("outer idle drive")
        .expect("nested foreground participation");
    assert_eq!(result, 7);
}

#[test]
fn child_session_participates_in_parent_poll_with_distinct_task_owner() {
    let parent = RuntimeLocalTaskSupervisor::new();
    let factory = parent.child_scope_factory();
    let child = factory.create().expect("parent admits child");
    let parent_id = owner();
    let child_id = owner();
    let parent_port = parent.submission_port(parent_id);
    let child_port = child.submission_port(child_id);
    assert!(!parent.owner_token().same(&child.owner_token()));
    assert!(child_port
        .spawn_local(parent_id, Box::pin(async {}))
        .is_err());
    let lease = child_port
        .spawn_local(child_id, Box::pin(async { tokio::task::yield_now().await }))
        .expect("child task");
    let outcome = host()
        .block_on(parent.drive_until(async {
            child
                .participate_foreground(lease.join())
                .await
                .expect("child uses active parent driver")
        }))
        .expect("parent drive");
    assert_eq!(outcome, RuntimeLocalTaskOutcome::Completed);
    child.shutdown();
    assert!(child_port
        .spawn_local(child_id, Box::pin(async {}))
        .is_err());
    let parent_lease = parent_port
        .spawn_local(parent_id, Box::pin(async {}))
        .expect("child close does not close parent");
    assert_eq!(
        host()
            .block_on(parent.drive_until(parent_lease.join()))
            .unwrap(),
        RuntimeLocalTaskOutcome::Completed
    );
}

#[test]
fn child_cannot_start_second_drive_while_parent_drive_is_pending() {
    let parent = RuntimeLocalTaskSupervisor::new();
    let child = parent
        .child_scope_factory()
        .create()
        .expect("child admitted");
    let mut idle = Box::pin(parent.drive_until(std::future::pending::<()>()));
    let wake = futures::task::noop_waker();
    let mut context = Context::from_waker(&wake);
    assert!(idle.as_mut().poll(&mut context).is_pending());
    assert!(futures::executor::block_on(child.participate_foreground(async {})).is_err());
    drop(idle);
    assert!(futures::executor::block_on(child.participate_foreground(async {})).is_ok());
}

#[test]
fn parent_retirement_cancels_child_tasks_and_revokes_weak_factory() {
    let parent = RuntimeLocalTaskSupervisor::new();
    let factory = parent.child_scope_factory();
    let child = factory.create().expect("child admitted");
    let child_id = owner();
    let port = child.submission_port(child_id);
    let drops = Rc::new(Cell::new(0));
    let lease = port
        .spawn_local(
            child_id,
            Box::pin(PendingDrop {
                drops: drops.clone(),
                _pinned: PhantomPinned,
            }),
        )
        .expect("pending child work");
    parent.shutdown();
    assert_eq!(drops.get(), 1);
    assert_eq!(lease.outcome(), Some(RuntimeLocalTaskOutcome::Cancelled));
    assert!(port.spawn_local(child_id, Box::pin(async {})).is_err());
    assert!(factory.create().is_err());
    drop(child);
    drop(parent);
    assert!(factory.create().is_err());
}

#[test]
fn weak_child_factory_does_not_retain_parent_or_accumulate_retired_children() {
    let parent = RuntimeLocalTaskSupervisor::new();
    let factory = parent.child_scope_factory();
    for _ in 0..3 {
        let child = factory.create().expect("child admitted");
        assert_eq!(parent.registered_child_count(), 1);
        drop(child);
        assert_eq!(parent.registered_child_count(), 0);
    }
    drop(parent);
    assert!(factory.create().is_err());
}

#[test]
fn callback_participates_under_the_exact_active_local_driver() {
    let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
    let owner = owner();
    let child_supervisor = Rc::downgrade(&supervisor);
    let lease = supervisor
        .submission_port(owner)
        .spawn_local(
            owner,
            Box::pin(async move {
                assert_eq!(
                    child_supervisor
                        .upgrade()
                        .expect("host still owns driver")
                        .participate_foreground(async { 11usize })
                        .await
                        .expect("callback belongs to this driver"),
                    11
                );
            }),
        )
        .expect("spawn callback");
    let outcome = host()
        .block_on(supervisor.drive_until(lease.join()))
        .expect("idle drive");
    assert_eq!(outcome, RuntimeLocalTaskOutcome::Completed);
    assert_eq!(
        supervisor.retained_task_count(),
        0,
        "completed record pruned despite retained lease"
    );
}

#[test]
fn pending_idle_drive_does_not_admit_unrelated_foreground() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let mut idle = Box::pin(supervisor.drive_until(std::future::pending::<()>()));
    let wake = futures::task::noop_waker();
    let mut context = Context::from_waker(&wake);
    assert!(idle.as_mut().poll(&mut context).is_pending());
    let polled = Rc::new(Cell::new(false));
    let witness = polled.clone();
    let unrelated = futures::executor::block_on(supervisor.participate_foreground(async move {
        witness.set(true);
    }));
    assert!(unrelated.is_err());
    assert!(!polled.get());
    drop(idle);
    assert!(futures::executor::block_on(supervisor.participate_foreground(async {})).is_ok());
}

#[test]
fn shutdown_releases_pending_child_even_when_drive_future_is_retained() {
    struct StartedPending {
        polls: Rc<Cell<usize>>,
        drops: Rc<Cell<usize>>,
        _pinned: PhantomPinned,
    }
    impl Future for StartedPending {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            self.polls.set(self.polls.get() + 1);
            Poll::Pending
        }
    }
    impl Drop for StartedPending {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let owner = owner();
    let drops = Rc::new(Cell::new(0));
    let polls = Rc::new(Cell::new(0));
    let lease = supervisor
        .submission_port(owner)
        .spawn_local(
            owner,
            Box::pin(StartedPending {
                polls: polls.clone(),
                drops: drops.clone(),
                _pinned: PhantomPinned,
            }),
        )
        .expect("schedule pending child");
    let mut idle = Box::pin(supervisor.drive_until(std::future::pending::<()>()));
    let wake = futures::task::noop_waker();
    let mut context = Context::from_waker(&wake);
    assert!(idle.as_mut().poll(&mut context).is_pending());
    assert!(polls.get() > 0, "child reached Pending before shutdown");
    supervisor.shutdown();
    assert_eq!(
        drops.get(),
        1,
        "host cancellation owns child drop without another poll"
    );
    assert_eq!(lease.outcome(), Some(RuntimeLocalTaskOutcome::Cancelled));
    assert_eq!(supervisor.retained_task_count(), 0);
    drop(idle);
    assert_eq!(drops.get(), 1);
}

#[test]
fn interleaved_pending_drivers_restore_exact_poll_owner() {
    let first = RuntimeLocalTaskSupervisor::new();
    let second = RuntimeLocalTaskSupervisor::new();
    let mut first_idle = Box::pin(first.drive_until(std::future::pending::<()>()));
    let mut second_idle = Box::pin(second.drive_until(std::future::pending::<()>()));
    let wake = futures::task::noop_waker();
    let mut context = Context::from_waker(&wake);
    assert!(first_idle.as_mut().poll(&mut context).is_pending());
    assert!(second_idle.as_mut().poll(&mut context).is_pending());
    assert!(futures::executor::block_on(first.participate_foreground(async {})).is_err());
    assert!(futures::executor::block_on(second.participate_foreground(async {})).is_err());
    drop(first_idle);
    assert!(futures::executor::block_on(first.participate_foreground(async {})).is_ok());
    assert!(futures::executor::block_on(second.participate_foreground(async {})).is_err());
    drop(second_idle);
    assert!(futures::executor::block_on(second.participate_foreground(async {})).is_ok());
}

#[test]
fn sibling_supervisor_cannot_enter_during_another_driver_poll() {
    let primary = RuntimeLocalTaskSupervisor::new();
    let sibling = RuntimeLocalTaskSupervisor::new();
    let result = futures::executor::block_on(
        primary.drive_until(async { sibling.participate_foreground(async { 1usize }).await }),
    )
    .expect("primary drive");
    assert!(result.is_err());
    assert_eq!(
        futures::executor::block_on(sibling.participate_foreground(async { 2usize })).unwrap(),
        2
    );
}

#[test]
fn panic_in_foreground_poll_clears_driver_owner_and_busy_state() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        futures::executor::block_on(supervisor.drive_until(async {
            panic!("foreground test unwind");
        }))
    }));
    assert!(result.is_err());
    assert!(futures::executor::block_on(supervisor.participate_foreground(async {})).is_ok());
}

#[test]
fn shutdown_from_a_polled_callback_does_not_borrow_or_drop_reentrantly() {
    let supervisor = Rc::new(RuntimeLocalTaskSupervisor::new());
    let owner = owner();
    let weak = Rc::downgrade(&supervisor);
    let (sender, receiver) = futures::channel::oneshot::channel();
    let lease = supervisor
        .submission_port(owner)
        .spawn_local(
            owner,
            Box::pin(async move {
                weak.upgrade().expect("host remains live").shutdown();
                let _ = sender.send(());
            }),
        )
        .expect("spawn");
    host()
        .block_on(supervisor.drive_until(async {
            receiver.await.expect("callback signalled after shutdown");
        }))
        .expect("same drive exits cleanly");
    assert_eq!(
        lease.outcome(),
        Some(RuntimeLocalTaskOutcome::Cancelled),
        "shutdown during the child's poll wins over its later Ready"
    );
    assert!(supervisor
        .submission_port(owner)
        .spawn_local(owner, Box::pin(async {}))
        .is_err());
}

#[test]
fn unpolled_and_pending_pinned_children_drop_once_on_shutdown() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let owner = owner();
    let port = supervisor.submission_port(owner);
    let drops = Rc::new(Cell::new(0));
    let first = port
        .spawn_local(
            owner,
            Box::pin(PendingDrop {
                drops: drops.clone(),
                _pinned: PhantomPinned,
            }),
        )
        .expect("unpolled task");
    let second = port
        .spawn_local(
            owner,
            Box::pin(PendingDrop {
                drops: drops.clone(),
                _pinned: PhantomPinned,
            }),
        )
        .expect("pending task");
    host()
        .block_on(supervisor.drive_until(async {
            tokio::task::yield_now().await;
        }))
        .expect("poll pending task");
    first.cancel();
    supervisor.shutdown();
    assert_eq!(drops.get(), 2);
    assert_eq!(first.outcome(), Some(RuntimeLocalTaskOutcome::Cancelled));
    assert_eq!(second.outcome(), Some(RuntimeLocalTaskOutcome::Cancelled));
    supervisor.shutdown();
    assert_eq!(drops.get(), 2);
}

#[test]
fn pending_task_does_not_keep_its_runtime_session_alive_after_shutdown() {
    let supervisor = RuntimeLocalTaskSupervisor::new();
    let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let owner = runtime.local_identity();
    let runtime = runtime.with_service_ports(
        RuntimeServicePorts::default().with_local_tasks(supervisor.submission_port(owner)),
    );
    let weak_session = Rc::downgrade(runtime.state());
    let drops = Rc::new(Cell::new(0));
    let task_context = runtime.clone();
    let lease = runtime
        .service_ports()
        .require_local_tasks("test timer")
        .unwrap()
        .spawn_local(
            owner,
            Box::pin(async move {
                task_context
                    .scope(PendingDrop {
                        drops,
                        _pinned: PhantomPinned,
                    })
                    .await;
            }),
        )
        .expect("spawn");
    // Timer state retains the lease while its callback retains the context.
    runtime
        .state()
        .resource_cleanup
        .0
        .borrow_mut()
        .push(Box::new(move || drop(lease)));
    drop(runtime);
    assert!(weak_session.upgrade().is_some());
    supervisor.shutdown();
    assert!(
        weak_session.upgrade().is_none(),
        "task must not cycle through its weak service port"
    );
}

mod deferred_admission;
