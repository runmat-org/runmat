use runmat_value::{CellArray, HandleRef, TransientValueError, Value};

use super::delta::validate_change_count;
use super::owner::validate_entry_count;
use super::*;

fn assigned(name: &str, value: Value) -> WorkspaceDelta {
    let mut delta = WorkspaceDelta::new();
    delta.assign(name, value).unwrap();
    delta
}

#[test]
fn ordered_delta_applies_clear_and_later_assignment_in_order() {
    let workspace = BaseWorkspaceOwner::new().unwrap();
    workspace
        .commit(
            WorkspaceRevision::INITIAL,
            assigned("before", Value::Num(1.0)),
        )
        .unwrap();

    let base = workspace.snapshot().unwrap().revision();
    let mut delta = WorkspaceDelta::new();
    delta.assign("discarded", Value::Num(2.0)).unwrap();
    delta.clear().unwrap();
    delta.assign("after", Value::Num(3.0)).unwrap();
    let commit = workspace.commit(base, delta).unwrap();

    assert_eq!(commit.revision().get(), 2);
    assert_eq!(commit.change_count(), 3);
    assert_eq!(
        commit.snapshot().values(),
        &std::collections::BTreeMap::from([("after".into(), Value::Num(3.0))])
    );
}

#[test]
fn projections_merge_unrelated_changes_and_later_same_name_write_wins() {
    let workspace = BaseWorkspaceOwner::new().unwrap();
    let mut first = workspace.begin_projection().unwrap();
    let mut second = workspace.begin_projection().unwrap();

    first.assign("left", Value::Num(1.0)).unwrap();
    first.assign("shared", Value::Num(10.0)).unwrap();
    assert!(!first.publish().unwrap().merged_concurrent_changes());

    second.assign("right", Value::Num(2.0)).unwrap();
    second.assign("shared", Value::Num(20.0)).unwrap();
    let commit = second.publish().unwrap();
    assert!(commit.merged_concurrent_changes());
    assert_eq!(commit.snapshot().get("left"), Some(&Value::Num(1.0)));
    assert_eq!(commit.snapshot().get("right"), Some(&Value::Num(2.0)));
    assert_eq!(commit.snapshot().get("shared"), Some(&Value::Num(20.0)));
}

#[test]
fn dirty_projection_must_publish_before_reconciliation() {
    let workspace = BaseWorkspaceOwner::new().unwrap();
    let mut projection = workspace.begin_projection().unwrap();
    projection.assign("local", Value::Num(1.0)).unwrap();
    workspace
        .commit(
            WorkspaceRevision::INITIAL,
            assigned("external", Value::Num(2.0)),
        )
        .unwrap();

    assert!(matches!(
        projection.reconcile(),
        Err(BaseWorkspaceError::PendingChanges)
    ));
    let publish = projection.publish().unwrap();
    assert!(publish.merged_concurrent_changes());
    assert_eq!(projection.lookup("external"), Some(&Value::Num(2.0)));
    assert_eq!(projection.lookup("local"), Some(&Value::Num(1.0)));
}

#[test]
fn clean_projection_reconciles_incrementally() {
    let workspace = BaseWorkspaceOwner::new().unwrap();
    let mut projection = workspace.begin_projection().unwrap();
    workspace
        .commit(
            WorkspaceRevision::INITIAL,
            assigned("external", Value::Num(4.0)),
        )
        .unwrap();

    let reconciliation = projection.reconcile().unwrap();
    assert!(matches!(
        reconciliation,
        BaseWorkspaceReconciliation::Delta {
            from: WorkspaceRevision::INITIAL,
            to,
            ..
        } if to.get() == 1
    ));
    assert_eq!(projection.lookup("external"), Some(&Value::Num(4.0)));
}

#[test]
fn bounded_history_returns_a_rooted_reset_snapshot() {
    let workspace = BaseWorkspaceOwner::for_test(1).unwrap();
    let mut projection = workspace.begin_projection().unwrap();
    workspace
        .commit(
            WorkspaceRevision::INITIAL,
            assigned("first", Value::Num(1.0)),
        )
        .unwrap();
    workspace
        .commit(
            WorkspaceRevision::from_raw(1),
            assigned("second", Value::Num(2.0)),
        )
        .unwrap();

    assert!(matches!(
        projection.reconcile().unwrap(),
        BaseWorkspaceReconciliation::Reset(_)
    ));
    assert_eq!(projection.lookup("first"), Some(&Value::Num(1.0)));
    assert_eq!(projection.lookup("second"), Some(&Value::Num(2.0)));
}

#[test]
fn revision_exhaustion_is_transactional() {
    let workspace = BaseWorkspaceOwner::new().unwrap();
    let exhausted = WorkspaceRevision::from_raw(u64::MAX);
    workspace.set_revision_for_test(exhausted);

    assert!(matches!(
        workspace.commit(exhausted, assigned("x", Value::Num(1.0))),
        Err(BaseWorkspaceError::RevisionExhausted)
    ));
    let snapshot = workspace.snapshot().unwrap();
    assert_eq!(snapshot.revision(), exhausted);
    assert!(snapshot.values().is_empty());
}

#[test]
fn transient_values_are_rejected_before_state_changes() {
    let workspace = BaseWorkspaceOwner::new().unwrap();
    let transient =
        Value::Cell(CellArray::new(vec![Value::OutputList(vec![Value::Num(1.0)])], 1, 1).unwrap());
    let mut delta = WorkspaceDelta::new();
    assert!(matches!(
        delta.assign("bad", transient),
        Err(BaseWorkspaceError::TransientValue(
            TransientValueError::LegacyOutputList
        ))
    ));
    assert!(workspace.snapshot().unwrap().values().is_empty());
}

#[test]
fn bounds_are_checked_without_large_allocations() {
    assert!(validate_change_count(MAX_WORKSPACE_DELTA_CHANGES).is_ok());
    assert!(matches!(
        validate_change_count(MAX_WORKSPACE_DELTA_CHANGES + 1),
        Err(BaseWorkspaceError::DeltaTooLarge { .. })
    ));
    assert!(validate_entry_count(MAX_BASE_WORKSPACE_ENTRIES).is_ok());
    assert!(matches!(
        validate_entry_count(MAX_BASE_WORKSPACE_ENTRIES + 1),
        Err(BaseWorkspaceError::TooManyEntries { .. })
    ));
}

#[test]
fn close_revokes_owner_and_projection_access() {
    let workspace = BaseWorkspaceOwner::new().unwrap();
    let mut projection = workspace.begin_projection().unwrap();
    projection.assign("pending", Value::Num(1.0)).unwrap();
    assert!(workspace.close().unwrap());
    assert!(!workspace.close().unwrap());
    assert!(workspace.is_closed().unwrap());
    assert!(matches!(
        workspace.snapshot(),
        Err(BaseWorkspaceError::Closed)
    ));
    assert!(matches!(
        projection.publish(),
        Err(BaseWorkspaceError::Closed)
    ));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn canonical_and_projection_values_hold_managed_roots_for_their_lifetimes() {
    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::String("retained".into())).unwrap();
        let handle = initial.handle();
        let value = Value::HandleObject(HandleRef {
            class_name: "WorkspaceHandle".into(),
            target: handle,
            valid: true,
        });
        let workspace = BaseWorkspaceOwner::new().unwrap();
        let mut projection = workspace.begin_projection().unwrap();
        projection.assign("held", value).unwrap();
        projection.publish().unwrap();
        initial.unroot().unwrap();

        runmat_gc::gc_collect_major().unwrap();
        assert_eq!(
            runmat_gc::gc_clone_value(&handle).unwrap(),
            Value::String("retained".into())
        );

        let revision = workspace.snapshot().unwrap().revision();
        let mut remove = WorkspaceDelta::new();
        remove.remove("held").unwrap();
        workspace.commit(revision, remove).unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());

        projection.reconcile().unwrap();
        drop(projection);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}
