use super::*;
use runmat_execution::profiling::{
    current_command_profiling, with_command_profiling, CommandProfilingRoot,
    CommandThreadCpuSampler, DeferredWorkCounts, ProviderObservationCounts,
    ProviderObservationTerminal,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

struct ObservedFuture {
    ready: bool,
    dropped: Arc<AtomicBool>,
}

impl Future for ObservedFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        assert!(
            current_command_profiling().is_some(),
            "inner poll selects captured origin"
        );
        if self.ready {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for ObservedFuture {
    fn drop(&mut self) {
        self.dropped
            .store(current_command_profiling().is_some(), Ordering::SeqCst);
    }
}

struct StepClock(std::sync::atomic::AtomicU64);

impl CommandThreadCpuSampler for StepClock {
    fn sample_current_thread(&self) -> Option<std::time::Duration> {
        let next = self.0.fetch_add(1, Ordering::SeqCst).checked_add(1)?;
        Some(std::time::Duration::from_nanos(next * 10))
    }
}

struct SealOnDrop {
    ready: bool,
    command: Option<CommandProfilingRoot>,
    observed: Arc<Mutex<Option<DeferredWorkCounts>>>,
}

impl Future for SealOnDrop {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
        if self.as_ref().get_ref().ready {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for SealOnDrop {
    fn drop(&mut self) {
        let snapshot = self.command.take().expect("one command").seal();
        *self.observed.lock().expect("drop observation") = snapshot.deferred_work;
    }
}

#[test]
fn inner_destructor_observes_pending_before_ready_or_unpolled_settlement() {
    for ready in [false, true] {
        let command = CommandProfilingRoot::new();
        let origin = command.receipt();
        let lease = origin.begin_deferred_work().expect("admission");
        let observed = Arc::new(Mutex::new(None));
        let mut task = Box::pin(RetiringNativeTask::new(
            SealOnDrop {
                ready,
                command: Some(command),
                observed: Arc::clone(&observed),
            },
            origin,
            lease,
        ));
        if ready {
            assert!(task
                .as_mut()
                .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
                .is_ready());
        }
        drop(task);
        assert_eq!(
            *observed.lock().expect("observed drop"),
            Some(DeferredWorkCounts {
                admitted: 1,
                pending: 1,
                ..DeferredWorkCounts::default()
            }),
            "inner future must retire before the token for ready={ready}"
        );
    }
}

#[test]
fn queued_unpolled_task_retires_inner_future_and_records_missing() {
    let clock = Arc::new(StepClock(std::sync::atomic::AtomicU64::new(0)));
    let command = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let dropped = Arc::new(AtomicBool::new(false));
    let task = RetiringNativeTask::new(
        ObservedFuture {
            ready: false,
            dropped: Arc::clone(&dropped),
        },
        command.receipt(),
        command.receipt().begin_deferred_work().unwrap(),
    );
    drop(task);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(clock.0.load(Ordering::SeqCst), 2);
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            missing: 1,
            ..DeferredWorkCounts::default()
        })
    );
}

#[test]
fn ready_task_retires_inner_future_before_completed_settlement() {
    let clock = Arc::new(StepClock(std::sync::atomic::AtomicU64::new(0)));
    let command = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let dropped = Arc::new(AtomicBool::new(false));
    let mut task = Box::pin(RetiringNativeTask::new(
        ObservedFuture {
            ready: true,
            dropped: Arc::clone(&dropped),
        },
        command.receipt(),
        command.receipt().begin_deferred_work().unwrap(),
    ));
    assert!(task
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
        .is_ready());
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(clock.0.load(Ordering::SeqCst), 2);
    drop(task);
    assert_eq!(
        clock.0.load(Ordering::SeqCst),
        2,
        "Ready already retired inner"
    );
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            completed: 1,
            ..DeferredWorkCounts::default()
        })
    );
}

#[test]
fn same_thread_caller_restores_after_pending_ready_and_inner_drop() {
    struct PollThenReady {
        polls: u8,
    }

    impl Future for PollThenReady {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            current_command_profiling()
                .expect("captured worker origin")
                .begin_provider_operation()
                .expect("worker command remains open")
                .complete(ProviderObservationTerminal::Missing);
            self.polls += 1;
            if self.polls == 1 {
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        }
    }

    impl Drop for PollThenReady {
        fn drop(&mut self) {
            current_command_profiling()
                .expect("captured origin through inner destructor")
                .begin_provider_operation()
                .expect("worker command remains open")
                .complete(ProviderObservationTerminal::Missing);
        }
    }

    let clock = Arc::new(StepClock(std::sync::atomic::AtomicU64::new(0)));
    let worker = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let caller = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let mut task = Box::pin(RetiringNativeTask::new(
        PollThenReady { polls: 0 },
        worker.receipt(),
        worker
            .receipt()
            .begin_deferred_work()
            .expect("worker admitted"),
    ));
    with_command_profiling(Some(caller.receipt()), || {
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        assert!(task.as_mut().poll(&mut context).is_pending());
        current_command_profiling()
            .expect("caller restored after Pending")
            .begin_provider_operation()
            .expect("caller still open")
            .complete(ProviderObservationTerminal::Missing);
        assert!(task.as_mut().poll(&mut context).is_ready());
        current_command_profiling()
            .expect("caller restored after Ready")
            .begin_provider_operation()
            .expect("caller still open")
            .complete(ProviderObservationTerminal::Missing);
        assert_eq!(clock.0.load(Ordering::SeqCst), 5);
        drop(task);
        assert_eq!(
            clock.0.load(Ordering::SeqCst),
            5,
            "no Ready Drop reselection"
        );
    });
    assert_eq!(clock.0.load(Ordering::SeqCst), 6);
    let worker = worker.seal();
    let caller = caller.seal();
    assert_eq!(
        worker.observations,
        Some(ProviderObservationCounts {
            missing: 3,
            ..ProviderObservationCounts::default()
        })
    );
    assert_eq!(
        caller.observations,
        Some(ProviderObservationCounts {
            missing: 2,
            ..ProviderObservationCounts::default()
        })
    );
    assert_eq!(
        worker.deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            completed: 1,
            ..DeferredWorkCounts::default()
        })
    );
    assert_eq!(
        worker.cpu_thread_attributed,
        Some(std::time::Duration::from_nanos(20))
    );
    assert_eq!(
        caller.cpu_thread_attributed,
        Some(std::time::Duration::from_nanos(30))
    );
}

#[test]
fn pending_worker_sealed_before_drop_keeps_immutable_partial_snapshot() {
    let clock = Arc::new(StepClock(std::sync::atomic::AtomicU64::new(0)));
    let command = CommandProfilingRoot::new_with_thread_cpu_sampler(Some(clock.clone()));
    let receipt = command.receipt();
    let dropped = Arc::new(AtomicBool::new(false));
    let mut task = Box::pin(RetiringNativeTask::new(
        ObservedFuture {
            ready: false,
            dropped: Arc::clone(&dropped),
        },
        receipt.clone(),
        receipt.begin_deferred_work().expect("admitted"),
    ));
    assert!(task
        .as_mut()
        .poll(&mut Context::from_waker(futures::task::noop_waker_ref()))
        .is_pending());
    let published = command.seal();
    assert_eq!(
        published.deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            pending: 1,
            ..DeferredWorkCounts::default()
        })
    );
    assert_eq!(
        published.cpu_thread_attributed,
        Some(std::time::Duration::from_nanos(10))
    );
    drop(task);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(
        clock.0.load(Ordering::SeqCst),
        2,
        "sealed owner makes no late clock reads"
    );
    assert!(receipt.begin_deferred_work().is_none());
    assert_eq!(published.deferred_work.expect("frozen").pending, 1);
    assert_eq!(
        published.cpu_thread_attributed,
        Some(std::time::Duration::from_nanos(10))
    );
}

#[test]
fn aborting_a_submitted_pending_task_settles_after_task_drop() {
    let driver = tokio::runtime::Runtime::new().expect("native driver");
    let command = CommandProfilingRoot::new();
    let dropped = Arc::new(AtomicBool::new(false));
    let task = spawn(
        driver.handle(),
        ObservedFuture {
            ready: false,
            dropped: Arc::clone(&dropped),
        },
        Some(command.receipt()),
    );
    task.abort();
    assert!(driver.block_on(task).is_err());
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            missing: 1,
            ..DeferredWorkCounts::default()
        })
    );
}

#[test]
fn panicking_task_destroys_inner_future_and_records_missing() {
    struct PanicOnPoll(Arc<AtomicBool>);

    impl Future for PanicOnPoll {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            panic!("native worker panic");
        }
    }

    impl Drop for PanicOnPoll {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let command = CommandProfilingRoot::new();
    let dropped = Arc::new(AtomicBool::new(false));
    let task = RetiringNativeTask::new(
        PanicOnPoll(Arc::clone(&dropped)),
        command.receipt(),
        command.receipt().begin_deferred_work().unwrap(),
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut task = Box::pin(task);
        let _ = task
            .as_mut()
            .poll(&mut Context::from_waker(futures::task::noop_waker_ref()));
    }))
    .is_err());
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(
        command.seal().deferred_work,
        Some(DeferredWorkCounts {
            admitted: 1,
            missing: 1,
            ..DeferredWorkCounts::default()
        })
    );
}
