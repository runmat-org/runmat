use super::*;
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
    assert!(registry.queue_explicit(entry));
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
        assert!(registry.queue_explicit(&entry));
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
        assert!(registry.queue_explicit(&entry));
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
