use super::*;
use runmat_execution::profiling::{
    CommandProfilingRoot, ProviderObservationCounts, ProviderObservationTerminal,
};
use runmat_value::ObjectInstance;

fn registered(registry: &CleanupRegistry) -> (GcHandle, Rc<CleanupEntry>) {
    let target = runmat_gc::gc_allocate_rooted(Value::Object(ObjectInstance::new(
        ON_CLEANUP_CLASS.to_string(),
    )))
    .expect("root target during registration");
    let handle = target.handle();
    let entry = registry
        .register(handle, Value::Num(1.0))
        .expect("register cleanup target");
    drop(target);
    (handle, entry)
}

fn complete(registry: &CleanupRegistry, entry: &Rc<CleanupEntry>) {
    assert!(registry.queue_explicit(entry, None));
    assert!(entry.begin_running());
    entry.complete(Ok(()));
    registry.mark_terminal(entry);
}

#[test]
fn many_completed_targets_retire_after_exact_gc_collection() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        for _ in 0..128 {
            let (_, entry) = registered(&registry);
            complete(&registry, &entry);
        }
        assert_eq!(registry.index.borrow().active.len(), 0);
        assert_eq!(registry.index.borrow().pending.len(), 0);
        runmat_gc::gc_collect_major().expect("collect released cleanup targets");
        registry.prune_terminal_after_collection();
        assert_eq!(registry.index.borrow().entries.len(), 0);
        assert_eq!(registry.index.borrow().terminal.len(), 0);
    });
}

#[test]
fn terminal_alias_stays_session_owned_until_its_exact_target_retires() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (target, entry) = registered(&registry);
        let alias = runmat_gc::gc_root(target).expect("external alias root");
        complete(&registry, &entry);
        runmat_gc::gc_collect_major().expect("collect with live alias");
        registry.prune_terminal_after_collection();
        assert!(registry.find(target).is_some());
        assert_eq!(entry.phase(), CleanupPhase::Completed);
        drop(alias);
        runmat_gc::gc_collect_major().expect("collect after alias release");
        registry.prune_terminal_after_collection();
        assert!(registry.find(target).is_none());
        // A caller already waiting on the entry keeps its receipt even
        // after the registry index releases the retired identity.
        assert!(matches!(
            entry.receipt(),
            Some(CleanupReceipt::Completed(Ok(())))
        ));
    });
}

#[test]
fn incomplete_retirement_probe_retries_without_rescanning_all_terminals() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (target, entry) = registered(&registry);
        let alias = runmat_gc::gc_root(target).expect("external alias root");
        complete(&registry, &entry);
        runmat_gc::gc_collect_major().expect("collect with live alias");
        runmat_gc::gc_with_value_mut(&target, |_| {
            registry.prune_terminal_after_collection();
            assert_eq!(registry.index.borrow().retirement_retry.len(), 1);
        })
        .expect("exclusive target borrow");
        registry.prune_terminal_after_collection();
        assert!(registry.index.borrow().retirement_retry.is_empty());
        assert!(registry.find(target).is_some());
        drop(alias);
        runmat_gc::gc_collect_major().expect("collect retired alias");
        registry.prune_terminal_after_collection();
        assert!(registry.find(target).is_none());
    });
}

#[test]
fn original_error_is_not_pruned_before_reporting() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (target, entry) = registered(&registry);
        assert!(registry.queue_explicit(&entry, None));
        assert!(entry.begin_running());
        entry.complete(Err(crate::runtime_error::semantic_error(
            "RunMat:onCleanup:OriginalFixture",
            "original callback failure",
        )));
        registry.mark_terminal(&entry);
        runmat_gc::gc_collect_major().expect("collect failed callback target");
        registry.prune_terminal_after_collection();
        assert!(registry.find(target).is_some());
        assert_eq!(registry.terminal_errors().len(), 1);
        assert!(entry.report_once());
        registry.note_reported(&entry);
        assert!(registry.find(target).is_none());
        let Some(CleanupReceipt::Completed(Err(error))) = entry.receipt() else {
            panic!("original typed error receipt was lost");
        };
        assert_eq!(
            crate::RuntimeError::from_shared_original(error).identifier(),
            Some("RunMat:onCleanup:OriginalFixture")
        );
    });
}

#[test]
fn reported_error_with_busy_target_retains_exact_retry_identity() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (target, entry) = registered(&registry);
        let alias = runmat_gc::gc_root(target).expect("external alias root");
        assert!(registry.queue_explicit(&entry, None));
        assert!(entry.begin_running());
        entry.complete(Err(crate::runtime_error::semantic_error(
            "RunMat:onCleanup:BusyFixture",
            "callback failed while target remains aliased",
        )));
        registry.mark_terminal(&entry);
        runmat_gc::gc_collect_major().expect("collect with live alias");
        assert!(entry.report_once());
        runmat_gc::gc_with_value_mut(&target, |_| {
            registry.note_reported(&entry);
            assert_eq!(registry.index.borrow().retirement_retry.len(), 1);
        })
        .expect("exclusive target borrow");
        registry.prune_terminal_after_collection();
        assert!(registry.index.borrow().retirement_retry.is_empty());
        assert!(registry.find(target).is_some());
        drop(alias);
        runmat_gc::gc_collect_major().expect("collect after alias release");
        registry.prune_terminal_after_collection();
        assert!(registry.find(target).is_none());
    });
}

#[test]
fn explicit_queue_keeps_the_admission_command_not_creation_or_later_drain() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let creation = CommandProfilingRoot::new();
        let (_target, entry) = registered(&registry);
        let admission = CommandProfilingRoot::new();
        let later_drain = CommandProfilingRoot::new();
        let admission_handle = admission.receipt();
        assert!(registry.queue_explicit(&entry, Some(&admission_handle)));
        assert!(!registry.queue_explicit(&entry, Some(&later_drain.receipt())));
        assert!(matches!(
            &*entry.deferred_origin.borrow(),
            CleanupDeferredOrigin::Admitted(_)
        ));
        entry
            .queued_command_origin()
            .expect("exact queue origin")
            .begin_provider_operation()
            .expect("admission still open")
            .complete(ProviderObservationTerminal::Missing);
        assert!(entry.begin_running());
        entry.complete(Ok(()));
        // This fixture directly completes an entry without a task; emulate
        // the owned callback future's subsequent destruction boundary.
        entry.settle_deferred_from_receipt();
        assert!(entry.queued_command_origin().is_none());
        assert_eq!(
            creation.seal().observations,
            Some(ProviderObservationCounts::default())
        );
        let admitted = admission.seal();
        assert_eq!(admitted.observations.expect("valid").missing, 1);
        assert_eq!(admitted.deferred_work.expect("valid deferred").completed, 1);
        assert_eq!(
            later_drain.seal().observations,
            Some(ProviderObservationCounts::default())
        );
    });
}

#[test]
fn cancelled_queue_releases_the_exact_command_origin_without_dispatch() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (_target, entry) = registered(&registry);
        let admission = CommandProfilingRoot::new();
        assert!(registry.queue_explicit(&entry, Some(&admission.receipt())));
        assert!(entry.queued_command_origin().is_some());
        assert!(registry.cancel(&entry));
        assert!(entry.queued_command_origin().is_none());
        let admitted = admission.seal();
        assert_eq!(
            admitted.observations,
            Some(ProviderObservationCounts::default())
        );
        assert_eq!(admitted.deferred_work.expect("valid deferred").missing, 1);
    });
}

#[test]
fn dropped_root_seals_a_queued_callback_origin_before_late_dispatch() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (_target, entry) = registered(&registry);
        let admission = CommandProfilingRoot::new();
        assert!(registry.queue_explicit(&entry, Some(&admission.receipt())));
        let borrowed = admission.receipt();
        let published = admission.seal();
        assert_eq!(published.deferred_work.expect("valid deferred").pending, 1);
        assert!(entry
            .queued_command_origin()
            .expect("queued origin retained until terminal")
            .begin_provider_operation()
            .is_none());
        assert!(registry.cancel(&entry));
        assert!(entry.queued_command_origin().is_none());
        assert!(borrowed.begin_deferred_work().is_none());
        assert_eq!(
            published.deferred_work.expect("immutable deferred").pending,
            1
        );
    });
}

#[test]
fn sealed_origin_denies_admission_without_becoming_an_unowned_zero() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (_target, entry) = registered(&registry);
        let command = CommandProfilingRoot::new();
        let borrowed = command.receipt();
        let published = command.seal();
        assert!(registry.queue_explicit(&entry, Some(&borrowed)));
        assert!(matches!(
            &*entry.deferred_origin.borrow(),
            CleanupDeferredOrigin::Unknown
        ));
        assert!(entry.queued_command_origin().is_some());
        assert!(registry.cancel(&entry));
        assert_eq!(
            published.deferred_work.expect("published counts").admitted,
            0
        );
        assert!(borrowed.begin_deferred_work().is_none());
    });
}

#[test]
fn chained_capture_release_admits_each_callback_under_its_actual_queue_origin() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (b_target, b) = registered(&registry);
        let a_target = runmat_gc::gc_allocate_rooted(Value::Object(ObjectInstance::new(
            ON_CLEANUP_CLASS.to_string(),
        )))
        .expect("root A target");
        let a = registry
            .register(
                a_target.handle(),
                Value::HandleObject(HandleRef {
                    class_name: ON_CLEANUP_CLASS.into(),
                    target: b_target,
                    valid: true,
                }),
            )
            .expect("A callback captures B target");
        drop(a_target);

        let first = CommandProfilingRoot::new();
        let CleanupAdmission::Queued(batch) = registry.admit_batch(&[], Some(&first.receipt()))
        else {
            panic!("A must be the first admissible callback");
        };
        assert_eq!(batch.len(), 1);
        assert!(Rc::ptr_eq(&batch[0], &a));
        assert_eq!(b.phase(), CleanupPhase::Active);
        assert!(a.begin_running());
        a.complete(Ok(()));
        registry.mark_terminal(&a);
        a.settle_deferred_from_receipt();
        let first_counts = first.seal().deferred_work.expect("A deferred valid");
        assert_eq!((first_counts.admitted, first_counts.completed), (1, 1));

        let second = CommandProfilingRoot::new();
        let CleanupAdmission::Queued(batch) = registry.admit_batch(&[], Some(&second.receipt()))
        else {
            panic!("B becomes eligible only after A callback capture retires");
        };
        assert_eq!(batch.len(), 1);
        assert!(Rc::ptr_eq(&batch[0], &b));
        assert!(b.begin_running());
        b.complete(Ok(()));
        registry.mark_terminal(&b);
        b.settle_deferred_from_receipt();
        let second_counts = second.seal().deferred_work.expect("B deferred valid");
        assert_eq!((second_counts.admitted, second_counts.completed), (1, 1));
    });
}

#[test]
fn synchronously_retired_rejected_submission_keeps_original_dispatch_failure() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (_target, entry) = registered(&registry);
        let command = CommandProfilingRoot::new();
        assert!(registry.queue_explicit(&entry, Some(&command.receipt())));
        entry.prepare_future_handoff();
        assert!(!entry.note_future_retired());
        assert!(entry.acknowledge_future_handoff(false));
        let error = crate::build_runtime_error("rejected without replay")
            .with_identifier("RunMat:onCleanup:ExactRejectedPort")
            .build();
        entry.dispatch_failed(error);
        registry.mark_terminal(&entry);
        assert!(matches!(
            entry.receipt(),
            Some(CleanupReceipt::DispatchFailed(_))
        ));
        assert_eq!(entry.phase(), CleanupPhase::DispatchFailed);
        assert!(!entry.test_has_callback_roots_and_deferred_admission());
        let error = match entry.receipt().expect("dispatch receipt") {
            CleanupReceipt::DispatchFailed(original) => original,
            _ => panic!("exact rejected receipt"),
        };
        assert_eq!(
            error.identifier(),
            Some("RunMat:onCleanup:ExactRejectedPort")
        );
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.failed), (1, 0, 1));
    });
}

#[test]
fn synchronously_retired_accepted_submission_is_missing_not_completed() {
    runmat_gc::gc_test_context(|| {
        let registry = CleanupRegistry::new();
        let (_target, entry) = registered(&registry);
        let command = CommandProfilingRoot::new();
        assert!(registry.queue_explicit(&entry, Some(&command.receipt())));
        entry.prepare_future_handoff();
        assert!(!entry.note_future_retired());
        assert!(entry.acknowledge_future_handoff(true));
        entry.abandon();
        registry.mark_terminal(&entry);
        entry.release_terminal_after_future_drop();
        assert_eq!(entry.phase(), CleanupPhase::Abandoned);
        let work = command.seal().deferred_work.expect("valid deferred work");
        assert_eq!((work.admitted, work.pending, work.missing), (1, 0, 1));
    });
}
