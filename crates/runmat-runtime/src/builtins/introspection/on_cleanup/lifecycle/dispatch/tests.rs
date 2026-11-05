use super::*;
use crate::builtins::introspection::on_cleanup::lifecycle::registry::CleanupPhase;
use futures::executor::block_on;
use runmat_execution::profiling::CommandProfilingRoot;
use runmat_value::{ObjectInstance, Value};
use std::cell::Cell;

struct ProbeFuture {
    entry: Rc<CleanupEntry>,
    dropped: Rc<Cell<bool>>,
    ready: bool,
    drop_phase: CleanupPhase,
    abandon_during_poll: Option<Rc<CleanupRegistry>>,
    panic_on_drop: bool,
}

impl Future for ProbeFuture {
    type Output = BuiltinResult<CleanupCallbackOutcome>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(registry) = &self.abandon_during_poll {
            registry.force_abandon();
        }
        if self.ready {
            Poll::Ready(Ok(CleanupCallbackOutcome::Completed))
        } else {
            Poll::Pending
        }
    }
}

impl Drop for ProbeFuture {
    fn drop(&mut self) {
        assert_eq!(self.entry.phase(), self.drop_phase);
        assert!(self.entry.test_has_callback_roots_and_deferred_admission());
        self.dropped.set(true);
        assert!(!self.panic_on_drop, "fixture resource Drop panicked");
    }
}

fn fixture() -> (
    RuntimeContext,
    Rc<CleanupRegistry>,
    Rc<CleanupEntry>,
    CommandProfilingRoot,
) {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let registry = Rc::new(CleanupRegistry::new());
    let target =
        runmat_gc::gc_allocate_rooted(Value::Object(ObjectInstance::new("onCleanup".to_string())))
            .expect("root target");
    let entry = registry
        .register(target.handle(), Value::Num(1.0))
        .expect("register callback");
    drop(target);
    let command = CommandProfilingRoot::new();
    assert!(registry.queue_explicit_for_completion(
        &entry,
        Some(&command.receipt()),
        Some(registry.issue_completion().unwrap()),
    ));
    entry.prepare_future_handoff();
    assert!(!entry.acknowledge_future_handoff(true));
    assert!(entry.begin_running());
    (runtime, registry, entry, command)
}

#[test]
fn scoped_completion_never_joins_an_unrelated_running_callback() {
    runmat_gc::gc_test_context(|| {
        use futures::FutureExt;
        let (runtime, registry, running, _command) = fixture();
        let completion = registry.issue_completion().expect("completion identity");
        let result = drain_completion(&runtime, &registry, completion)
            .now_or_never()
            .expect("unrelated running callback cannot suspend this completion");
        assert!(result.is_ok());
        assert_eq!(running.phase(), CleanupPhase::Running);
        let another = registry.issue_completion().expect("distinct completion");
        assert_ne!(completion, another);
        // A second independent callback's empty completion is equally ready;
        // neither callback can create a mutual wait on the other's task.
        assert!(drain_completion(&runtime, &registry, another)
            .now_or_never()
            .expect("independent completion")
            .is_ok());
        running.complete(Err(crate::build_runtime_error(
            "unrelated original failure",
        )
        .with_identifier("Cleanup:Unrelated")
        .build()));
        registry.mark_terminal(&running);
        assert!(drain_completion(&runtime, &registry, completion)
            .now_or_never()
            .expect("foreign terminal error cannot suspend")
            .is_ok());
        assert_eq!(
            registry.terminal_errors().len(),
            1,
            "unrelated failure is still unreported"
        );
        assert_eq!(
            receipt_result(&registry, &running)
                .unwrap_err()
                .identifier(),
            Some("Cleanup:Unrelated")
        );
    });
}

#[test]
fn ready_retires_inner_captures_before_roots_and_deferred_success() {
    runmat_gc::gc_test_context(|| {
        let (runtime, registry, entry, command) = fixture();
        let dropped = Rc::new(Cell::new(false));
        block_on(OwnedCleanupFuture {
            inner: Some(Box::pin(ProbeFuture {
                entry: Rc::clone(&entry),
                dropped: Rc::clone(&dropped),
                ready: true,
                drop_phase: CleanupPhase::Running,
                abandon_during_poll: None,
                panic_on_drop: false,
            })),
            entry: Rc::clone(&entry),
            registry: Some(registry),
            runtime: Some(runtime),
        });
        assert!(dropped.get());
        assert_eq!(entry.phase(), CleanupPhase::Completed);
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.completed), (1, 0, 1));
    });
}

#[test]
fn active_pending_drop_retires_inner_then_abandons_and_records_missing() {
    runmat_gc::gc_test_context(|| {
        let (runtime, registry, entry, command) = fixture();
        let dropped = Rc::new(Cell::new(false));
        let mut owned = Box::pin(OwnedCleanupFuture {
            inner: Some(Box::pin(ProbeFuture {
                entry: Rc::clone(&entry),
                dropped: Rc::clone(&dropped),
                ready: false,
                drop_phase: CleanupPhase::Running,
                abandon_during_poll: None,
                panic_on_drop: false,
            })),
            entry: Rc::clone(&entry),
            registry: Some(registry),
            runtime: Some(runtime),
        });
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(owned.as_mut().poll(&mut context).is_pending());
        drop(owned);
        assert!(dropped.get());
        assert_eq!(entry.phase(), CleanupPhase::Abandoned);
        assert!(entry.test_callback_root_released());
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.missing), (1, 0, 1));
    });
}

#[test]
fn reentrant_host_cancellation_during_poll_keeps_roots_until_inner_drop() {
    runmat_gc::gc_test_context(|| {
        let (runtime, registry, entry, command) = fixture();
        let drain_runtime = runtime.clone();
        let dropped = Rc::new(Cell::new(false));
        let mut owned = Box::pin(OwnedCleanupFuture {
            inner: Some(Box::pin(ProbeFuture {
                entry: Rc::clone(&entry),
                dropped: Rc::clone(&dropped),
                ready: false,
                drop_phase: CleanupPhase::Abandoned,
                abandon_during_poll: Some(Rc::clone(&registry)),
                panic_on_drop: false,
            })),
            entry: Rc::clone(&entry),
            registry: Some(Rc::clone(&registry)),
            runtime: Some(runtime),
        });
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(owned.as_mut().poll(&mut context).is_pending());
        assert_eq!(entry.phase(), CleanupPhase::Abandoned);
        assert!(entry.test_has_callback_roots_and_deferred_admission());
        let mut drain = Box::pin(drain_completion(
            &drain_runtime,
            &registry,
            entry.completion().unwrap(),
        ));
        assert!(
            drain.as_mut().poll(&mut context).is_pending(),
            "cancel acknowledgement cannot publish before actual future retirement"
        );
        drop(owned);
        assert!(matches!(
            drain.as_mut().poll(&mut context),
            Poll::Ready(Err(_))
        ));
        assert!(dropped.get());
        assert!(entry.test_callback_root_released());
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.missing), (1, 0, 1));
    });
}

#[test]
fn inner_resource_drop_unwind_cannot_strand_callback_roots_or_deferred_work() {
    runmat_gc::gc_test_context(|| {
        let (runtime, registry, entry, command) = fixture();
        let dropped = Rc::new(Cell::new(false));
        let owned = OwnedCleanupFuture {
            inner: Some(Box::pin(ProbeFuture {
                entry: Rc::clone(&entry),
                dropped: Rc::clone(&dropped),
                ready: false,
                drop_phase: CleanupPhase::Running,
                abandon_during_poll: None,
                panic_on_drop: true,
            })),
            entry: Rc::clone(&entry),
            registry: Some(registry),
            runtime: Some(runtime),
        };
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(owned)));
        assert!(unwind.is_err());
        assert!(dropped.get());
        assert_eq!(entry.phase(), CleanupPhase::Abandoned);
        assert!(entry.test_callback_root_released());
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.missing), (1, 0, 1));
    });
}
