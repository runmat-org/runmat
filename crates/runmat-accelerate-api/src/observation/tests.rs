use super::*;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll, Wake, Waker};

#[derive(Default)]
struct TestClock(AtomicU64);
impl TestClock {
    fn set(&self, nanos: u64) {
        self.0.store(nanos, Ordering::SeqCst);
    }
}
impl ProviderObservationClock for TestClock {
    fn now(&self) -> Option<Duration> {
        Some(Duration::from_nanos(self.0.load(Ordering::SeqCst)))
    }
}

#[test]
fn completion_and_publication_both_required_total_excludes_drain_delay() {
    let clock = Arc::new(TestClock::default());
    clock.set(10);
    let mut receipt = ProviderOperationReceipt::with_clock(clock.clone());
    let recorder = receipt.recorder();
    let first = recorder.submission().unwrap();
    let second = recorder.submission().unwrap();
    clock.set(15);
    receipt.publish().unwrap();
    clock.set(20);
    first.complete();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    clock.set(30);
    std::thread::spawn(move || second.complete())
        .join()
        .unwrap();
    clock.set(1000);
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Completed {
            total_elapsed: Duration::from_nanos(20),
            submissions: 2
        }
    );
    assert!(receipt.take_completed().is_none());
}

#[test]
fn publication_after_device_completion_and_empty_operation() {
    let clock = Arc::new(TestClock::default());
    let mut receipt = ProviderOperationReceipt::with_clock(clock.clone());
    let submitted = receipt.recorder().submission().unwrap();
    clock.set(4);
    submitted.complete();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    clock.set(9);
    receipt.publish().unwrap();
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Completed {
            total_elapsed: Duration::from_nanos(9),
            submissions: 1
        }
    );
    let mut empty = ProviderOperationReceipt::with_clock(clock.clone());
    clock.set(10);
    empty.publish().unwrap();
    assert_eq!(
        empty.take_completed().unwrap().status,
        ProviderObservationStatus::Completed {
            total_elapsed: Duration::from_nanos(1),
            submissions: 0
        }
    );
}

#[test]
fn cancellation_failure_abandonment_and_late_callbacks_never_publish_success() {
    for failure in [
        ProviderObservationFailure::Cancelled,
        ProviderObservationFailure::PublicationFailed,
    ] {
        let mut receipt = ProviderOperationReceipt::new();
        let completion = receipt.recorder().submission().unwrap();
        receipt.fail(failure);
        completion.complete();
        assert_eq!(
            receipt.take_completed().unwrap().status,
            ProviderObservationStatus::Failed(failure)
        );
    }
    let mut receipt = ProviderOperationReceipt::new();
    receipt.recorder().submission().unwrap().fail();
    assert_eq!(
        receipt.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ProviderFailed)
    );
    assert!(receipt.take_completed().is_some());
    let receipt = ProviderOperationReceipt::new();
    let recorder = receipt.recorder();
    let completion = recorder.submission().unwrap();
    drop(receipt);
    completion.complete();
    assert!(matches!(
        recorder.submission(),
        Err(ProviderObservationFailure::Abandoned)
    ));
    let mut receipt = ProviderOperationReceipt::new();
    drop(receipt.recorder().submission().unwrap());
    assert!(receipt.publish().is_err());
}

#[test]
fn late_registration_fails_closed_and_stages_do_not_replace_total() {
    let clock = Arc::new(TestClock::default());
    let mut receipt = ProviderOperationReceipt::with_clock(clock.clone());
    let recorder = receipt.recorder();
    recorder.record_stage(ProviderObservationStage::Upload, Duration::from_nanos(8));
    recorder.record_stage(ProviderObservationStage::Execution, Duration::from_nanos(8));
    recorder.record_pipeline(runmat_execution::CandidatePreparationState::Cold);
    recorder.record_pipeline(runmat_execution::CandidatePreparationState::Warm);
    clock.set(10);
    receipt.publish().unwrap();
    let result = receipt.take_completed().unwrap();
    assert_eq!(
        result.status,
        ProviderObservationStatus::Completed {
            total_elapsed: Duration::from_nanos(10),
            submissions: 0
        }
    );
    assert_eq!(result.pipelines.cold, 1);
    assert_eq!(result.pipelines.warm, 1);
    assert_eq!(
        result.stages[ProviderObservationStage::Upload as usize].elapsed,
        Duration::from_nanos(8)
    );
    let mut receipt = ProviderOperationReceipt::new();
    receipt.publish().unwrap();
    assert!(receipt.recorder().submission().is_err());
    assert_eq!(
        receipt.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::PublicationFailed)
    );
}

struct NoClock;
impl ProviderObservationClock for NoClock {
    fn now(&self) -> Option<Duration> {
        None
    }
}

#[test]
fn missing_or_invalid_timing_never_becomes_positive_feedback() {
    let mut unavailable = ProviderOperationReceipt::with_clock(Arc::new(NoClock));
    assert!(unavailable.publish().is_err());
    let clock = Arc::new(TestClock::default());
    clock.set(10);
    let mut receipt = ProviderOperationReceipt::with_clock(clock.clone());
    let token = receipt.recorder().submission().unwrap();
    clock.set(9);
    token.complete();
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::InvalidTiming)
    );
}

struct Noop;
impl Wake for Noop {
    fn wake(self: Arc<Self>) {}
}

#[test]
fn nested_interleaved_pending_ready_and_unwind_restore_activation() {
    let mut outer = ProviderOperationReceipt::new();
    let mut a = ProviderOperationReceipt::new();
    let mut b = ProviderOperationReceipt::new();
    let mut polls = 0;
    let mut future_a = Box::pin(observe_provider_future(
        Some(a.recorder()),
        std::future::poll_fn(|_| {
            current_provider_observation()
                .unwrap()
                .submission()
                .unwrap()
                .complete();
            polls += 1;
            if polls == 1 {
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        }),
    ));
    let mut future_b = Box::pin(observe_provider_future(Some(b.recorder()), async {
        current_provider_observation()
            .unwrap()
            .submission()
            .unwrap()
            .complete();
    }));
    let waker = Waker::from(Arc::new(Noop));
    let mut cx = Context::from_waker(&waker);
    with_provider_observation(Some(outer.recorder()), || {
        assert!(future_a.as_mut().poll(&mut cx).is_pending());
        assert!(future_b.as_mut().poll(&mut cx).is_ready());
        assert!(future_a.as_mut().poll(&mut cx).is_ready());
        let result = std::panic::catch_unwind(|| {
            with_provider_observation(None, || {
                assert!(current_provider_observation().is_none());
                panic!("injected poll panic");
            });
        });
        assert!(result.is_err());
        current_provider_observation()
            .unwrap()
            .submission()
            .unwrap()
            .complete();
    });
    assert!(current_provider_observation().is_none());
    for (receipt, expected) in [(&mut outer, 1), (&mut a, 2), (&mut b, 1)] {
        receipt.publish().unwrap();
        assert!(
            matches!(receipt.take_completed().unwrap().status, ProviderObservationStatus::Completed { submissions, .. } if submissions == expected)
        );
    }
}

struct ReentrantWake {
    recorder: ProviderOperationRecorder,
    calls: Arc<AtomicU64>,
}
impl Wake for ReentrantWake {
    fn wake(self: Arc<Self>) {
        // Reentering the receipt would deadlock if callbacks ran under its lock.
        self.recorder
            .record_stage(ProviderObservationStage::Queue, Duration::ZERO);
        self.calls.fetch_add(1, Ordering::SeqCst);
    }
}
impl Drop for ReentrantWake {
    fn drop(&mut self) {
        self.recorder
            .record_stage(ProviderObservationStage::Queue, Duration::ZERO);
    }
}

#[test]
fn completion_wakes_owner_without_lock_held_and_poll_is_single_delivery() {
    let mut receipt = ProviderOperationReceipt::new();
    let submission = receipt.recorder().submission().unwrap();
    receipt.publish().unwrap();
    let calls = Arc::new(AtomicU64::new(0));
    let waker = Waker::from(Arc::new(ReentrantWake {
        recorder: receipt.recorder(),
        calls: calls.clone(),
    }));
    let mut cx = Context::from_waker(&waker);
    assert!(receipt.poll_completed(&mut cx).is_pending());
    std::thread::spawn(move || submission.complete())
        .join()
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        receipt.poll_completed(&mut cx),
        Poll::Ready(Some(_))
    ));
    assert_eq!(receipt.poll_completed(&mut cx), Poll::Ready(None));
}

#[test]
fn dropping_unpolled_or_pending_future_does_not_leave_activation() {
    let receipt = ProviderOperationReceipt::new();
    drop(observe_provider_future(
        Some(receipt.recorder()),
        std::future::pending::<()>(),
    ));
    assert!(current_provider_observation().is_none());
    let mut future = Box::pin(observe_provider_future(
        Some(receipt.recorder()),
        std::future::pending::<()>(),
    ));
    let waker = Waker::from(Arc::new(Noop));
    assert!(future
        .as_mut()
        .poll(&mut Context::from_waker(&waker))
        .is_pending());
    drop(future);
    assert!(current_provider_observation().is_none());
    let recorder = receipt.recorder();
    drop(receipt); // The invocation owner, not poll activation, owns abort.
    assert!(matches!(
        recorder.submission(),
        Err(ProviderObservationFailure::Abandoned)
    ));
}

#[test]
fn abandoning_receipt_releases_retained_waiter_cycle() {
    let mut receipt = ProviderOperationReceipt::new();
    let task = Arc::new(ReentrantWake {
        recorder: receipt.recorder(),
        calls: Arc::new(AtomicU64::new(0)),
    });
    let weak = Arc::downgrade(&task);
    let waker = Waker::from(task);
    assert!(receipt
        .poll_completed(&mut Context::from_waker(&waker))
        .is_pending());
    drop(waker);
    assert!(weak.upgrade().is_some());
    drop(receipt);
    assert!(weak.upgrade().is_none());
}

#[test]
fn duplicate_publication_fails_before_delivery_but_cannot_revoke_delivery() {
    let mut receipt = ProviderOperationReceipt::new();
    receipt.publish().unwrap();
    assert_eq!(
        receipt.publish(),
        Err(ProviderObservationFailure::PublicationFailed)
    );
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Failed(ProviderObservationFailure::PublicationFailed)
    );

    let mut receipt = ProviderOperationReceipt::new();
    receipt.publish().unwrap();
    let delivered = receipt.take_completed().unwrap();
    assert!(matches!(
        delivered.status,
        ProviderObservationStatus::Completed { .. }
    ));
    assert_eq!(
        receipt.publish(),
        Err(ProviderObservationFailure::PublicationFailed)
    );
    assert_eq!(receipt.status(), delivered.status);
    assert!(receipt.take_completed().is_none());
}

#[test]
fn measurements_freeze_at_completion_not_delayed_delivery() {
    use runmat_execution::CandidatePreparationState;
    let mut receipt = ProviderOperationReceipt::new();
    let recorder = receipt.recorder();
    let submission = recorder.submission().unwrap();
    receipt.publish().unwrap();
    recorder.record_stage(ProviderObservationStage::Execution, Duration::from_nanos(7));
    recorder.record_pipeline(CandidatePreparationState::Warm);
    submission.complete();
    recorder.record_stage(
        ProviderObservationStage::Execution,
        Duration::from_nanos(99),
    );
    recorder.record_pipeline(CandidatePreparationState::Cold);
    let evidence = receipt.take_completed().unwrap();
    let execution = evidence.stages[ProviderObservationStage::Execution as usize];
    assert_eq!(execution.samples, 1);
    assert_eq!(execution.elapsed, Duration::from_nanos(7));
    assert_eq!(evidence.pipelines.warm, 1);
    assert_eq!(evidence.pipelines.cold, 0);
}
