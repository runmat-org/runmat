mod disposition;
mod stochastic_transfer;

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

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn only_default_clock_receipts_certify_the_outer_prelude() {
    let outer = runmat_time::Instant::now();
    let default = ProviderOperationReceipt::new();
    assert!(default.default_clock_prelude_since(outer).is_some());
    let budget = Arc::new(ProviderObservationBudget::new(
        std::num::NonZeroU32::new(1).unwrap(),
    ));
    let budgeted = ProviderOperationReceipt::with_budget(budget);
    assert!(budgeted.default_clock_prelude_since(outer).is_some());
    let custom = ProviderOperationReceipt::with_clock(Arc::new(TestClock::default()));
    assert!(custom.default_clock_prelude_since(outer).is_none());
}

#[test]
fn verified_wgpu_fields_require_every_physical_submission_and_settle_once() {
    let mut receipt = ProviderOperationReceipt::new();
    let first = receipt
        .recorder()
        .submission()
        .unwrap()
        .capture_completion();
    let second = receipt
        .recorder()
        .submission()
        .unwrap()
        .capture_completion();
    receipt.publish().unwrap();
    first.complete_with_wgpu(WgpuSubmissionMeasurement {
        device_elapsed: Some(Duration::from_nanos(13)),
        compute_elapsed: Some(Duration::from_nanos(7)),
        dispatches: Some(2),
        query_allocation: Some(WgpuQueryAllocationDisposition::Allocated {
            timestamp_slots: 16,
            resolve_requested_bytes: 32,
            staging_requested_bytes: 32,
            map_callbacks: 1,
        }),
    });
    second.complete_with_wgpu(WgpuSubmissionMeasurement {
        device_elapsed: None,
        compute_elapsed: Some(Duration::from_nanos(11)),
        dispatches: Some(1),
        query_allocation: Some(WgpuQueryAllocationDisposition::Denied),
    });
    let observed = receipt.take_completed().unwrap();
    assert_eq!(
        observed.verified_wgpu_compute_elapsed,
        Some(Duration::from_nanos(18))
    );
    assert_eq!(observed.verified_wgpu_dispatches, Some(3));
    assert_eq!(
        observed.wgpu_query_allocations,
        Some(WgpuQueryAllocationObservation {
            query_sets: 1,
            timestamp_slots: 16,
            resolve_requested_bytes: 32,
            staging_requested_bytes: 32,
            map_callbacks: 1,
            denied_submissions: 1,
            ..WgpuQueryAllocationObservation::default()
        })
    );
    assert!(receipt.take_completed().is_none());

    let mut mixed = ProviderOperationReceipt::new();
    let measured = mixed.recorder().submission().unwrap().capture_completion();
    let unmeasured = mixed.recorder().submission().unwrap().capture_completion();
    mixed.publish().unwrap();
    measured.complete_with_wgpu(WgpuSubmissionMeasurement {
        device_elapsed: None,
        compute_elapsed: Some(Duration::from_nanos(9)),
        dispatches: Some(1),
        query_allocation: None,
    });
    unmeasured.complete();
    let observed = mixed.take_completed().unwrap();
    assert_eq!(observed.verified_wgpu_compute_elapsed, None);
    assert_eq!(observed.verified_wgpu_dispatches, None);
    assert_eq!(observed.wgpu_query_allocations, None);
}

#[test]
fn unsupported_timestamp_preserves_exact_dispatches_but_empty_receipt_is_unknown() {
    let mut receipt = ProviderOperationReceipt::new();
    let completed = receipt
        .recorder()
        .submission()
        .unwrap()
        .capture_completion();
    receipt.publish().unwrap();
    completed.complete_with_wgpu(WgpuSubmissionMeasurement {
        device_elapsed: None,
        compute_elapsed: None,
        dispatches: Some(0),
        query_allocation: Some(WgpuQueryAllocationDisposition::NoComputePass),
    });
    let observed = receipt.take_completed().unwrap();
    assert_eq!(observed.verified_wgpu_compute_elapsed, None);
    assert_eq!(observed.verified_wgpu_dispatches, Some(0));
    assert_eq!(
        observed
            .wgpu_query_allocations
            .unwrap()
            .no_compute_submissions,
        1
    );

    let mut empty = ProviderOperationReceipt::new();
    empty.publish().unwrap();
    let observed = empty.take_completed().unwrap();
    assert_eq!(observed.verified_wgpu_compute_elapsed, None);
    assert_eq!(observed.verified_wgpu_dispatches, None);
}

#[test]
fn take_and_poll_publish_the_same_independent_wgpu_evidence() {
    struct NoopWake;
    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    let sample = WgpuSubmissionMeasurement {
        compute_elapsed: None,
        dispatches: Some(2),
        query_allocation: Some(WgpuQueryAllocationDisposition::Denied),
        ..WgpuSubmissionMeasurement::default()
    };
    let mut observations = Vec::new();
    for poll in [false, true] {
        let mut receipt = ProviderOperationReceipt::with_clock(Arc::new(TestClock::default()));
        let completed = receipt
            .recorder()
            .submission()
            .unwrap()
            .capture_completion();
        receipt.publish().unwrap();
        completed.complete_with_wgpu(sample);
        let observed = if poll {
            let waker = Waker::from(Arc::new(NoopWake));
            let mut cx = Context::from_waker(&waker);
            let Poll::Ready(Some(observed)) = receipt.poll_completed(&mut cx) else {
                panic!("completed receipt must poll ready")
            };
            observed
        } else {
            receipt.take_completed().expect("completed receipt")
        };
        observations.push(observed);
    }
    assert_eq!(observations[0], observations[1]);
    assert_eq!(observations[0].verified_wgpu_compute_elapsed, None);
    assert_eq!(observations[0].verified_wgpu_dispatches, Some(2));
    assert_eq!(
        observations[0]
            .wgpu_query_allocations
            .unwrap()
            .denied_submissions,
        1
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn query_bytes_and_required_submission_slots_admit_independently_and_refund() {
    let budget = Arc::new(ProviderObservationBudget::with_wgpu_query_bytes(
        std::num::NonZeroU32::new(1).unwrap(),
        256,
    ));
    assert!(budget.try_reserve_wgpu_query(0).is_none());
    assert!(budget.try_reserve_wgpu_query(257).is_none());
    let query = budget.try_reserve_wgpu_query(256).expect("query admitted");
    assert!(budget.try_reserve_wgpu_query(1).is_none());

    let receipt = ProviderOperationReceipt::with_budget(budget.clone());
    let submission = receipt
        .recorder()
        .submission()
        .expect("required submission unaffected");
    assert!(receipt.recorder().submission().is_err());
    assert!(budget.try_reserve_wgpu_query(1).is_none());
    drop(query);
    let next_query = budget
        .try_reserve_wgpu_query(256)
        .expect("query bytes refunded");
    submission.fail();
    drop(receipt);

    let next = ProviderOperationReceipt::with_budget(budget.clone());
    let next_submission = next
        .recorder()
        .submission()
        .expect("submission slot refunded");
    assert!(budget.try_reserve_wgpu_query(1).is_none());
    next_submission.fail();
    drop(next_query);
    assert!(budget.try_reserve_wgpu_query(256).is_some());
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn poisoned_optional_query_budget_rejects_new_admission_but_not_required_work() {
    let budget = Arc::new(ProviderObservationBudget::with_wgpu_query_bytes(
        std::num::NonZeroU32::new(1).unwrap(),
        256,
    ));
    let held = budget.try_reserve_wgpu_query(128).expect("first query");
    let before = Arc::new(std::sync::Barrier::new(2));
    let after = Arc::new(std::sync::Barrier::new(2));
    let other_budget = budget.clone();
    let worker_before = before.clone();
    let worker_after = after.clone();
    let worker = std::thread::spawn(move || {
        worker_before.wait();
        let raced = other_budget.try_reserve_wgpu_query(128);
        worker_after.wait();
        raced
    });
    before.wait();
    // The racing reservation may have linearized before poison. In either
    // order, no admission can linearize after this monotone transition.
    held.poison_budget();
    after.wait();
    let raced = worker.join().unwrap();
    drop(raced);
    drop(held);
    assert!(budget.try_reserve_wgpu_query(1).is_none());
    let receipt = ProviderOperationReceipt::with_budget(budget);
    let required = receipt.recorder().submission().expect("required work");
    required.fail();
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
fn callback_time_is_preserved_while_cleanup_keeps_evidence_pending() {
    let clock = Arc::new(TestClock::default());
    let mut receipt = ProviderOperationReceipt::with_clock(clock.clone());
    let submitted = receipt.recorder().submission().unwrap();
    clock.set(4);
    receipt.publish().unwrap();
    clock.set(9);
    let completed = submitted.capture_completion();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    clock.set(1000);
    completed.complete();
    assert_eq!(
        receipt.take_completed().unwrap().status,
        ProviderObservationStatus::Completed {
            total_elapsed: Duration::from_nanos(9),
            submissions: 1,
        }
    );
    clock.set(2000);
    let mut later_publication = ProviderOperationReceipt::with_clock(clock.clone());
    let submitted = later_publication.recorder().submission().unwrap();
    clock.set(2004);
    let completed = submitted.capture_completion();
    clock.set(2009);
    later_publication.publish().unwrap();
    clock.set(3000);
    completed.complete();
    assert_eq!(
        later_publication.take_completed().unwrap().status,
        ProviderObservationStatus::Completed {
            total_elapsed: Duration::from_nanos(9),
            submissions: 1,
        }
    );
}

#[test]
fn deferred_cleanup_failure_drop_and_cancellation_cannot_publish_success() {
    for mode in 0..3 {
        let mut receipt = ProviderOperationReceipt::new();
        let completed = receipt
            .recorder()
            .submission()
            .unwrap()
            .capture_completion();
        receipt.publish().unwrap();
        match mode {
            0 => completed.fail(),
            1 => drop(completed),
            _ => {
                receipt.fail(ProviderObservationFailure::Cancelled);
                completed.complete();
            }
        }
        let expected = match mode {
            0 => ProviderObservationFailure::ProviderFailed,
            1 => ProviderObservationFailure::Abandoned,
            _ => ProviderObservationFailure::Cancelled,
        };
        assert_eq!(
            receipt.take_completed().unwrap().status,
            ProviderObservationStatus::Failed(expected)
        );
    }
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
        result.stages[ProviderObservationStage::Upload.index()].elapsed,
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

#[test]
fn provider_host_mediation_needs_its_own_post_upload_submission() {
    let operation = crate::ProviderOperationIdentity::new("imfilter");
    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_selected_execution();
    receipt.require_provider_execution(7, operation.clone());
    receipt.allow_provider_host_mediated();
    let recorder = receipt.recorder();
    let preupload = recorder.submission().unwrap();
    preupload.complete();
    let checkpoint = recorder.submission_checkpoint();
    recorder.confirm_provider_host_mediated_after(7, &operation, checkpoint);
    assert_eq!(
        receipt.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ExecutionUnproven)
    );

    let mut receipt = ProviderOperationReceipt::new();
    receipt.require_selected_execution();
    receipt.require_provider_execution(7, operation.clone());
    receipt.allow_provider_host_mediated();
    let recorder = receipt.recorder();
    let checkpoint = recorder.submission_checkpoint();
    let completion_fence = recorder.submission().unwrap();
    recorder.confirm_provider_host_mediated_after(7, &operation, checkpoint);
    receipt.publish().unwrap();
    assert_eq!(receipt.status(), ProviderObservationStatus::Pending);
    completion_fence.complete();
    let completed = receipt.take_completed().unwrap();
    assert!(matches!(
        completed.status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
    assert_eq!(
        completed.route_outcome,
        Some(ProviderRouteOutcome::ProviderHostMediated)
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
    let execution = evidence.stages[ProviderObservationStage::Execution.index()];
    assert_eq!(execution.samples, 1);
    assert_eq!(execution.elapsed, Duration::from_nanos(7));
    assert_eq!(evidence.pipelines.warm, 1);
    assert_eq!(evidence.pipelines.cold, 0);
}

#[test]
fn provider_budget_is_shared_and_cancellation_does_not_recycle_live_tokens() {
    let budget = Arc::new(ProviderObservationBudget::new(
        std::num::NonZeroU32::new(1).unwrap(),
    ));
    let mut first = ProviderOperationReceipt::with_budget(budget.clone());
    let callback = first.recorder().submission().unwrap();
    first.fail(ProviderObservationFailure::Cancelled);
    drop(first);
    let second = ProviderOperationReceipt::with_budget(budget.clone());
    assert!(matches!(
        second.recorder().submission(),
        Err(ProviderObservationFailure::ResourceLimit)
    ));
    assert_eq!(
        second.status(),
        ProviderObservationStatus::Failed(ProviderObservationFailure::ResourceLimit)
    );
    callback.complete();
    let mut third = ProviderOperationReceipt::with_budget(budget);
    third.recorder().submission().unwrap().complete();
    third.publish().unwrap();
    assert!(matches!(
        third.take_completed().unwrap().status,
        ProviderObservationStatus::Completed { submissions: 1, .. }
    ));
}

#[test]
fn every_stage_has_a_distinct_bounded_measurement_slot() {
    let mut receipt = ProviderOperationReceipt::new();
    for stage in ProviderObservationStage::ALL {
        receipt
            .recorder()
            .record_stage(stage, Duration::from_nanos(stage.index() as u64 + 1));
    }
    receipt.publish().unwrap();
    let observation = receipt.take_completed().unwrap();
    for stage in ProviderObservationStage::ALL {
        assert_eq!(observation.stages[stage.index()].samples, 1);
        assert_eq!(
            observation.stages[stage.index()].elapsed,
            Duration::from_nanos(stage.index() as u64 + 1)
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn outstanding_callback_does_not_keep_progress_sender_alive() {
    struct Driver;
    impl WgpuObservationProgress for Driver {
        fn reserve(
            &self,
        ) -> Result<Box<dyn WgpuObservationProgressReservation>, WgpuObservationProgressError>
        {
            Err(WgpuObservationProgressError::Stopped)
        }
    }
    let driver = Arc::new(Driver);
    let weak = Arc::downgrade(&driver);
    let budget = Arc::new(ProviderObservationBudget::new(
        std::num::NonZeroU32::new(1).unwrap(),
    ));
    let outer = runmat_time::Instant::now();
    let receipt = ProviderOperationReceipt::with_wgpu_progress(budget, driver.clone());
    assert!(receipt.default_clock_prelude_since(outer).is_some());
    let callback = receipt.recorder().submission().unwrap();
    drop(receipt);
    drop(driver);
    assert!(
        weak.upgrade().is_none(),
        "map callback must not retain worker sender"
    );
    callback.complete();
}

#[test]
fn abandoned_future_cleanup_uses_its_owner_not_ambient_parent() {
    struct Cleanup;
    impl Future for Cleanup {
        type Output = ();
        fn poll(self: std::pin::Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            Poll::Pending
        }
    }
    impl Drop for Cleanup {
        fn drop(&mut self) {
            current_provider_observation()
                .unwrap()
                .submission()
                .unwrap()
                .complete();
        }
    }
    for polled in [false, true] {
        let mut outer = ProviderOperationReceipt::new();
        let mut inner = ProviderOperationReceipt::new();
        let mut future = Box::pin(observe_provider_future(Some(inner.recorder()), Cleanup));
        with_provider_observation(Some(outer.recorder()), || {
            if polled {
                let waker = Waker::from(Arc::new(Noop));
                assert!(future
                    .as_mut()
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending());
            }
            drop(future);
        });
        inner.publish().unwrap();
        outer.publish().unwrap();
        assert!(matches!(
            inner.take_completed().unwrap().status,
            ProviderObservationStatus::Completed { submissions: 1, .. }
        ));
        assert!(matches!(
            outer.take_completed().unwrap().status,
            ProviderObservationStatus::Completed { submissions: 0, .. }
        ));
    }
}
